"""
Local GGUF reranker powered by llama-cpp-python.

This reranker uses repeated pairwise comparisons instead of one large
listwise prompt because small local models are often overly sensitive to
input ordering in listwise ranking tasks.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from video_config import LLM_MODEL_PATH, LLM_N_CTX, LLM_N_GPU_LAYERS, TOP_K
from video_router.intent_router import build_chatml_prompt, load_llama_model

_SYSTEM_PROMPT = """You are a precise reranker for a movie scene retrieval system.
You compare two candidates and choose the one that better matches the user query.
You MUST respond with a single valid JSON object and nothing else."""

_MAX_TEXT_CHARS = 2000


class LLMReranker:
    """Rerank fused retrieval candidates with the local GGUF LLM."""

    def __init__(
        self,
        model_path: str = LLM_MODEL_PATH,
        n_ctx: int = LLM_N_CTX,
        n_gpu_layers: int = LLM_N_GPU_LAYERS,
        llm: Any | None = None,
        backend: str | None = None,
    ) -> None:
        if llm is not None:
            self._llm = llm
            self._backend = backend or "cpu"
            print(f"[LLMReranker] Using shared model instance on backend: {self._backend}")
        else:
            self._llm, self._backend, _ = load_llama_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                component_name="LLMReranker",
            )

    def rerank(
        self,
        user_query: str,
        results: list[dict],
        intent: dict | None = None,
        top_k: int = TOP_K,
    ) -> list[dict]:
        """
        Return the same results in LLM-ranked order.

        Falls back to the original retrieval order whenever a pairwise
        comparison is undecidable or the LLM output cannot be parsed.
        """
        if not results:
            return []

        candidates = [
            {
                "candidate_id": _build_candidate_id(result),
                "description": _truncate(result.get("description", "")),
                "content": _truncate(result.get("content", "")),
                "result": result.copy(),
            }
            for result in results
        ]

        comparison_cache: dict[tuple[str, str], int] = {}
        ordered: list[dict] = []

        # Stable insertion sort using pairwise LLM comparisons.
        # Ties preserve the original retrieval order.
        for candidate in candidates:
            insert_at = len(ordered)
            for idx, existing in enumerate(ordered):
                comparison = self._compare_candidates(
                    user_query=user_query,
                    left=candidate,
                    right=existing,
                    cache=comparison_cache,
                )
                if comparison < 0:
                    insert_at = idx
                    break
            ordered.insert(insert_at, candidate)

        reranked: list[dict] = []
        for llm_rank, candidate in enumerate(ordered[:top_k], start=1):
            entry = candidate["result"].copy()
            entry["llm_rerank_rank"] = llm_rank
            reranked.append(entry)

        return reranked

    def _compare_candidates(
        self,
        user_query: str,
        left: dict,
        right: dict,
        cache: dict[tuple[str, str], int],
    ) -> int:
        """
        Compare two candidates.

        Returns:
            -1 if left should rank ahead of right
             1 if right should rank ahead of left
             0 if undecidable, so caller should preserve retrieval order
        """
        left_id = left["candidate_id"]
        right_id = right["candidate_id"]
        cache_key = tuple(sorted((left_id, right_id)))

        if cache_key in cache:
            cached = cache[cache_key]
            if cache_key[0] == left_id:
                return cached
            return -cached

        winner_forward = self._pick_winner(user_query, left, right)
        winner_reverse = self._pick_winner(user_query, right, left)

        if winner_forward == left_id and winner_reverse == left_id:
            result = -1
        elif winner_forward == right_id and winner_reverse == right_id:
            result = 1
        else:
            result = 0

        if cache_key[0] == left_id:
            cache[cache_key] = result
        else:
            cache[cache_key] = -result
        return result

    def _pick_winner(
        self,
        user_query: str,
        left: dict,
        right: dict,
    ) -> str | None:
        prompt = self._build_prompt(
            user_query=user_query,
            left_candidate=left,
            right_candidate=right,
        )

        try:
            raw_output = self._call_llm(prompt)
            return self._parse_winner_id(
                raw_output,
                valid_ids={left["candidate_id"], right["candidate_id"]},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[LLMReranker] Pairwise comparison failed ({exc}), preserving retrieval order.")
            return None

    def _build_prompt(
        self,
        user_query: str,
        left_candidate: dict,
        right_candidate: dict,
    ) -> str:
        left_payload = {
            "candidate_id": left_candidate["candidate_id"],
            "description": left_candidate["description"],
            "content": left_candidate["content"],
        }
        right_payload = {
            "candidate_id": right_candidate["candidate_id"],
            "description": right_candidate["description"],
            "content": right_candidate["content"],
        }

        user_prompt = (
            "Choose the single candidate that better matches the user query.\n\n"
            "Rules:\n"
            "- Use only description and content as evidence.\n"
            "- Return exactly one winner_id.\n"
            "- If the candidates seem close, choose the better match anyway.\n\n"
            f"User query: {user_query}\n"
            f"Candidate A: {json.dumps(left_payload, ensure_ascii=False, separators=(',', ':'))}\n"
            f"Candidate B: {json.dumps(right_payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
            'Respond ONLY with JSON in this exact schema:\n{"winner_id":"<candidate_id>"}'
        )
        return build_chatml_prompt(_SYSTEM_PROMPT, user_prompt)

    def _call_llm(self, prompt: str) -> str:
        self._llm.reset()
        response = self._llm(
            prompt,
            max_tokens=64,
            temperature=0.0,
            stop=["<|im_end|>", "\n\n\n"],
        )
        return response["choices"][0]["text"].strip()

    def _parse_winner_id(self, raw_output: str, valid_ids: set[str]) -> str:
        payload = _extract_json(raw_output)

        if not isinstance(payload, dict):
            raise ValueError("Pairwise reranker response was not a JSON object.")

        winner_id = payload.get("winner_id")
        if not isinstance(winner_id, str) or winner_id not in valid_ids:
            raise ValueError(f"Invalid winner_id in reranker output: {raw_output!r}")

        return winner_id


def _extract_json(raw_output: str) -> Any:
    """Parse direct JSON first, then extract the first object/array block."""
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\}|\[.*\])", raw_output, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in reranker output: {raw_output!r}")

    return json.loads(match.group(1))


def _truncate(text: str, max_chars: int = _MAX_TEXT_CHARS) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


def _build_candidate_id(result: dict) -> str:
    """Create an opaque id that does not leak retrieval rank to the LLM."""
    source = f"{result.get('movie', '')}::{result.get('clip_id', '')}"
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    return f"cand_{digest}"
