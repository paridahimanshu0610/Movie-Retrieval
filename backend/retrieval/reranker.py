"""
Screenplay result reranker powered by TAMU AI (Claude Sonnet 4.5).

Sends all candidates in a single batch call and asks the LLM to return
a ranked list of candidate IDs. One LLM call regardless of result count,
compared to the O(n²) calls of pairwise insertion sort.

Falls back to RRF order silently if TAMU_AI_CHAT_API_KEY is not set or
if the LLM returns an unparseable response.
"""

from __future__ import annotations

import json
import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()

TAMU_AI_ENDPOINT = "https://chat-api.tamu.ai/api/v1/chat/completions"
TAMU_AI_MODEL    = "protected.Claude Sonnet 4.5"

_SYSTEM_PROMPT = """You are a precise reranker for a movie identification system.
Given a user query and a list of candidate movies with matching scene evidence, rank the candidates from most to least relevant.
Base your ranking only on the matching_scenes evidence provided — not on general knowledge about the movies.
You MUST respond with a single valid JSON object and nothing else."""

_MAX_SCENE_CHARS = 300


class ScreenplayReranker:
    """Rerank screenplay retrieval results via a single TAMU AI batch call."""

    def __init__(self, registry: dict) -> None:
        self._api_key = os.environ.get("TAMU_AI_CHAT_API_KEY")
        self._registry = registry
        if self._api_key:
            print("[ScreenplayReranker] Using TAMU AI Chat for reranking.")
        else:
            print("[ScreenplayReranker] TAMU_AI_CHAT_API_KEY not set — using RRF order.")

    def rerank(self, query: str, raw_results: list[dict], top_k: int) -> list[dict]:
        """
        Rerank raw retriever results (list of {movie_id, score, top_scenes}).
        Returns the same list reordered, trimmed to top_k.
        """
        if not self._api_key or len(raw_results) <= 1:
            return raw_results[:top_k]

        candidates = [self._build_candidate(r) for r in raw_results[:top_k]]
        id_to_result = {c["candidate_id"]: c["result"] for c in candidates}

        ranked_ids = self._batch_rank(query, candidates)
        if not ranked_ids:
            return raw_results[:top_k]

        # Build reranked list; append any candidates the LLM omitted at the end
        seen = set()
        reranked = []
        for cid in ranked_ids:
            if cid in id_to_result and cid not in seen:
                reranked.append(id_to_result[cid])
                seen.add(cid)
        for r in raw_results[:top_k]:
            if r["movie_id"] not in seen:
                reranked.append(r)

        return reranked[:top_k]

    def _build_candidate(self, result: dict) -> dict:
        movie_id = result["movie_id"]
        info     = self._registry.get(movie_id, {})
        title    = info.get("title", movie_id)
        year     = info.get("year", "")

        scenes = []
        for scene in result.get("top_scenes", [])[:3]:
            heading = scene.get("heading", "")
            text    = _truncate(scene.get("text", scene.get("overview", "")))
            if heading or text:
                scenes.append({"heading": heading, "text": text})

        return {
            "candidate_id":   movie_id,
            "title":          title,
            "year":           year,
            "matching_scenes": scenes,
            "result":         result,
        }

    def _batch_rank(self, query: str, candidates: list[dict]) -> list[str] | None:
        payload = [
            {
                "candidate_id":    c["candidate_id"],
                "title":           c["title"],
                "year":            c["year"],
                "matching_scenes": c["matching_scenes"],
            }
            for c in candidates
        ]
        valid_ids = {c["candidate_id"] for c in candidates}

        user_prompt = (
            "Rank the following movie candidates from most to least relevant to the user query.\n\n"
            "Rules:\n"
            "- Use only the matching_scenes as evidence.\n"
            "- Return all candidate_ids in ranked order.\n\n"
            f"User query: {query}\n"
            f"Candidates: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
            'Respond ONLY with JSON: {"ranked_ids": ["<candidate_id>", ...]}'
        )

        try:
            response = requests.post(
                TAMU_AI_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       TAMU_AI_MODEL,
                    "stream":      False,
                    "messages":    [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "max_tokens":  2048,
                    "temperature": 1,
                },
                timeout=60,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"].strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
            ranked_ids = json.loads(raw).get("ranked_ids", [])
            return [cid for cid in ranked_ids if cid in valid_ids]
        except Exception as exc:
            print(f"[ScreenplayReranker] Batch ranking failed ({exc}), using RRF order.")
            return None


def _truncate(text: str, max_chars: int = _MAX_SCENE_CHARS) -> str:
    cleaned = " ".join(str(text).split())
    return cleaned if len(cleaned) <= max_chars else cleaned[:max_chars - 1].rstrip() + "…"
