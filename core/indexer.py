"""
indexer.py — Dual-index storage and Reciprocal Rank Fusion retrieval.

Structure
─────────
DualIndex        : stores visual + text embeddings + metadata for every clip
cosine_search    : returns ranked (clip_idx, score) pairs for a query vector
reciprocal_rank_fusion : merges two ranked lists into a single RRF-scored list
Retriever        : high-level search interface
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import config


# ── Dual Index ────────────────────────────────────────────────────────────────

class DualIndex:
    """
    In-memory index with numpy arrays.
    For production, swap the numpy cosine search for a FAISS IVF index.
    """

    def __init__(self):
        self._visual: List[np.ndarray] = []   # visual embeddings
        self._text:   List[np.ndarray] = []   # text   embeddings
        self.metadata: List[Dict]      = []

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add(
        self,
        clip_id:    str,
        movie:      str,
        visual_emb: np.ndarray,
        text_emb:   np.ndarray,
        caption:    str,
        transcript: str,
        path:       str,
    ) -> None:
        self._visual.append(_l2norm(visual_emb))
        self._text.append(_l2norm(text_emb))
        self.metadata.append(
            dict(
                clip_id=clip_id,
                movie=movie,
                caption=caption,
                transcript=transcript[:300],
                path=path,
            )
        )

    def __len__(self) -> int:
        return len(self.metadata)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "visual.npy", np.array(self._visual, dtype=np.float32))
        np.save(directory / "text.npy",   np.array(self._text,   dtype=np.float32))
        with open(directory / "metadata.json", "w") as fh:
            json.dump(self.metadata, fh, indent=2, ensure_ascii=False)
        print(f"[DualIndex] Saved {len(self)} clips → {directory}")

    def load(self, directory: Path) -> None:
        visual_arr = np.load(directory / "visual.npy")
        text_arr   = np.load(directory / "text.npy")
        self._visual   = list(visual_arr)
        self._text     = list(text_arr)
        with open(directory / "metadata.json") as fh:
            self.metadata = json.load(fh)
        print(f"[DualIndex] Loaded {len(self)} clips from {directory}")

    # ── Search (internal) ─────────────────────────────────────────────────────

    def search_visual(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        return _cosine_search(query_vec, self._visual, top_k)

    def search_text(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        return _cosine_search(query_vec, self._text, top_k)


# ── Search helpers ────────────────────────────────────────────────────────────

def _l2norm(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / (n + 1e-8)


def _cosine_search(
    query: np.ndarray,
    corpus: List[np.ndarray],
    top_k: int,
) -> List[Tuple[int, float]]:
    """
    Brute-force cosine similarity.
    For >10 k clips, replace with FAISS:
        import faiss
        index = faiss.IndexFlatIP(dim)
        index.add(np.array(corpus))
        scores, ids = index.search(query[None], top_k)
    """
    q = _l2norm(query)
    mat = np.array(corpus, dtype=np.float32)   # (N, D)
    scores = mat @ q                            # (N,)
    top = int(min(top_k, len(scores)))
    ranked_idx = np.argpartition(-scores, top - 1)[:top]
    ranked_idx = ranked_idx[np.argsort(-scores[ranked_idx])]
    return [(int(i), float(scores[i])) for i in ranked_idx]


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[int, float]]],
    k: int = config.RRF_K,
) -> List[Tuple[int, float]]:
    """
    Standard RRF: score(d) = Σ  1 / (k + rank(d))
    Returns a merged list sorted by descending RRF score.
    """
    rrf: Dict[int, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(rrf.items(), key=lambda x: -x[1])


# ── High-level Retriever ──────────────────────────────────────────────────────

class Retriever:
    """
    Given a natural-language query, searches both indexes in parallel,
    merges with RRF, and returns ranked metadata records.
    """

    def __init__(self, index: DualIndex, text_embedder, visual_embedder):
        self._index           = index
        self._text_embedder   = text_embedder    # Sentence-BERT
        self._visual_embedder = visual_embedder  # InternVideo2 / CLIP

    def search(self, query: str, top_k: int = 5, debug: bool = False) -> List[Dict]:
        if len(self._index) == 0:
            raise RuntimeError("Index is empty. Run build_index.py first.")

        vis_q_vec  = self._visual_embedder.embed_text(query)  # aligned with visual index
        text_q_vec = self._text_embedder.embed(query)         # aligned with text index

        vis_results  = self._index.search_visual(vis_q_vec,  top_k * 3)
        text_results = self._index.search_text  (text_q_vec, top_k * 3)
        
        if debug==True:
            # Checking the results for visual search
            for idx, score in vis_results:
                metadata = self._index.metadata[idx]
                print(f"Visual Search - Clip ID: {metadata['clip_id']}, Movie: {metadata['movie']}, Score: {score}")
            
            # Checking the results for text search
            for idx, score in text_results:
                metadata = self._index.metadata[idx]
                print(f"Text Search - Clip ID: {metadata['clip_id']}, Movie: {metadata['movie']}, Score: {score}")

        # Merge with RRF
        merged = reciprocal_rank_fusion([vis_results, text_results])

        results: List[Dict] = []
        for doc_id, rrf_score in merged[:top_k]:
            rec = dict(self._index.metadata[doc_id])
            rec["rrf_score"] = round(rrf_score, 5)
            # Individual sub-index scores for diagnostics
            vis_score  = next((s for i, s in vis_results  if i == doc_id), 0.0)
            text_score = next((s for i, s in text_results if i == doc_id), 0.0)
            rec["visual_score"] = round(float(vis_score), 4)
            rec["text_score"]   = round(float(text_score), 4)
            results.append(rec)

        return results