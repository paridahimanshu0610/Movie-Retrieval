"""
BM25 retriever for dialogue/transcript queries.

Scores are raw BM25 values (not normalised) — they are only used for
within-list ranking before RRF merges them with dense scores.
"""

from __future__ import annotations

from config import TOP_K
from indexer.bm25_indexer import BM25Indexer, tokenize


class BM25Retriever:
    def __init__(self, index: BM25Indexer) -> None:
        if index.bm25 is None:
            raise ValueError("BM25Indexer has not been built/loaded yet.")
        self.index = index

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Retrieve top-k clips ranked by BM25 score against their transcripts.

        Args:
            query:  User query (or rewritten query from the intent router).
            top_k:  Maximum results to return.

        Returns:
            List of result dicts (same shape as ChromaRetriever output)
            sorted by BM25 score descending.  Clips with score == 0 are
            excluded (no term overlap).
        """
        tokens = tokenize(query)
        scores = self.index.bm25.get_scores(tokens)

        # Rank by score descending, keep top-k with non-zero score
        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in ranked:
            if scores[idx] == 0.0:
                break  # all remaining will also be 0 (sorted)
            meta = self.index.corpus_meta[idx]
            results.append(
                {
                    "doc_id":       f"{meta['movie']}__{meta['clip_id']}__dialogue",
                    "clip_id":      meta["clip_id"],
                    "movie":        meta["movie"],
                    "field_type":   "dialogue",
                    "score":        round(float(scores[idx]), 4),
                    "content":      self.index.corpus_texts[idx],
                    "youtube_link": meta.get("youtube_link", ""),
                    "timestamp":    meta.get("timestamp", ""),
                    "description":  meta.get("description", ""),
                }
            )

        return results