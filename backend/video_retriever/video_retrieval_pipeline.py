"""
RetrievalPipeline — the single public entry point for a search query.

Data flow
---------
User query
    │
    ▼
IntentRouter  →  { primary_field, secondary_fields, rewritten_query}
    │
    ├── [primary_field != dialogue OR always]  →  ChromaRetriever(primary_field)
    │
    └── [secondary_fields present]  →  ChromaRetriever(secondary_fields)
                │
                ▼
        Reciprocal Rank Fusion
                │
                ▼
        Local LLM rerank
                │
                ▼
        Deduplicated top-K results

Modes
-----
The pipeline can be initialised in two modes:

  "full"    → IntentRouter (LLM) + ChromaRetriever + BM25Retriever
  "no_llm"  → keyword_fallback + ChromaRetriever + BM25Retriever
              (useful for testing or when the GGUF model is unavailable)
"""

from __future__ import annotations

from threading import RLock

from video_config import BM25_INDEX_PATH, LLM_MODEL_PATH, LLM_N_GPU_LAYERS, TOP_K
from video_indexer.bm25_indexer import BM25Indexer
from video_indexer.embedder import Embedder
from video_retriever.bm25_retriever import BM25Retriever
from video_retriever.chroma_retriever import ChromaRetriever
from video_retriever.fusion import reciprocal_rank_fusion
from video_retriever.llm_reranker import LLMReranker
from video_router.intent_router import IntentRouter, keyword_fallback, load_llama_model


class RetrievalPipeline:
    def __init__(
        self,
        llm_model_path: str = LLM_MODEL_PATH,
        n_gpu_layers: int = LLM_N_GPU_LAYERS,
        use_llm: bool = True,
    ) -> None:
        """
        Args:
            llm_model_path: Absolute path to the .gguf file.
            n_gpu_layers:   GPU layers to offload (0 = CPU only).
            use_llm:        Set False to skip LLM initialisation and use the
                            keyword fallback router instead.
        """
        print("[Pipeline] Initialising …")

        self._embedder = Embedder()
        self._chroma = ChromaRetriever(self._embedder)
        self._query_lock = RLock()

        bm25_index = BM25Indexer.load(BM25_INDEX_PATH)
        self._bm25 = BM25Retriever(bm25_index)

        self._router: IntentRouter | None = None
        self._reranker: LLMReranker | None = None

        if use_llm:
            shared_llm, backend, _ = load_llama_model(
                model_path=llm_model_path,
                n_gpu_layers=n_gpu_layers,
                component_name="Pipeline",
            )
            self._router = IntentRouter(
                llm=shared_llm,
                backend=backend,
            )
            self._reranker = LLMReranker(
                llm=shared_llm,
                backend=backend,
            )
        else:
            print("[Pipeline] LLM disabled — using keyword fallback router.")

        print("[Pipeline] Ready.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def query(self, user_query: str, top_k: int = TOP_K) -> dict:
        """
        Run a full retrieval pipeline for *user_query*.

        Args:
            user_query: Raw query string from the user.
            top_k:      Number of final results to return.

        Returns a dict with:
            query         : original query
            intent        : dict from the router
            results       : list of top_k result dicts (see below)

        Each result dict contains:
            clip_id, movie, field_type, score, rrf_score,
            matched_fields, youtube_link, timestamp, description, content,
            retrieval_rank, llm_rerank_rank
        """
        # The shared llama.cpp instance is reused across router + reranker,
        # so keep full-query execution single-threaded when the pipeline is
        # shared by a long-lived FastAPI process.
        with self._query_lock:
            return self._query_unlocked(user_query=user_query, top_k=top_k)

    def _query_unlocked(self, user_query: str, top_k: int = TOP_K) -> dict:
        """Internal query implementation. Callers should hold `_query_lock`."""
        # ── Step 1: Intent extraction ──────────────────────────────────────────
        if self._router is not None:
            intent = self._router.route(user_query)
        else:
            intent = keyword_fallback(user_query)

        print(
            f"[Pipeline] Intent: primary={intent['primary_field']} "
            f"secondary={intent['secondary_fields']} "
        )

        rewritten = intent["rewritten_query"]
        result_lists: list[list[dict]] = []

        # ── Step 2a: BM25 for dialogue ─────────────────────────────────────────
        if intent["primary_field"] == "dialogue":
            bm25_hits = self._bm25.retrieve(user_query, top_k=top_k) # use original query for BM25 to preserve verbatim matching
            if bm25_hits:
                result_lists.append(bm25_hits)

        # ── Step 2b: Dense — primary field ────────────────────────────────────
        # Always run dense for the primary field (even if BM25 also ran)
        # unless the primary field IS dialogue and BM25 already covered it.
        if not (intent["primary_field"] == "dialogue"):
            primary_hits = self._chroma.retrieve(
                rewritten,
                field_types=[intent["primary_field"]],
                top_k=top_k,
            )
            if primary_hits:
                result_lists.append(primary_hits)

        # ── Step 2c: Dense — secondary fields ─────────────────────────────────
        if intent["secondary_fields"]:
            if "dialogue" in intent["secondary_fields"]:
                bm25_hits = self._bm25.retrieve(user_query, top_k=top_k) # use original query for BM25 to preserve verbatim matching
                if bm25_hits:
                    result_lists.append(bm25_hits)
                # Remove dialogue from secondary_fields to avoid duplicate BM25 retrieval
                intent["secondary_fields"] = [ft for ft in intent["secondary_fields"] if ft != "dialogue"]

            secondary_hits = self._chroma.retrieve(
                rewritten,
                field_types=intent["secondary_fields"],
                top_k=top_k,
            )
            if secondary_hits:
                result_lists.append(secondary_hits)

        # ── Step 3: Fuse ───────────────────────────────────────────────────────
        if len(result_lists) > 1:
            fused = reciprocal_rank_fusion(result_lists, top_k=None)
        elif len(result_lists) == 1:
            # Single list: add dummy rrf_score + matched_fields for consistent shape
            fused = []
            for r in result_lists[0][:top_k]:
                r = r.copy()
                r["rrf_score"] = r.get("score", 0.0)
                r["matched_fields"] = [r.get("field_type", "")]
                fused.append(r)
        else:
            fused = []

        fused = _annotate_retrieval_rank(fused)
        print(f"[Pipeline] Fused {len(fused)} results.")
        # ── Step 4: Local LLM rerank ──────────────────────────────────────────
        # Keep the rerank window at top_k so the prompt stays compact enough
        # for the small local context window configured for llama.cpp.
        if self._reranker is not None and fused:
            final_results = self._reranker.rerank(
                user_query=user_query,
                results=fused[:top_k],
                intent=intent,
                top_k=top_k,
            )
        else:
            final_results = fused[:top_k]

        return {
            "query": user_query,
            "intent": intent,
            "results": final_results,
        }


def _annotate_retrieval_rank(results: list[dict]) -> list[dict]:
    """Preserve the pre-rerank ordering for transparency and debugging."""
    annotated: list[dict] = []
    for rank, result in enumerate(results, start=1):
        entry = result.copy()
        entry["retrieval_rank"] = rank
        annotated.append(entry)
    return annotated
