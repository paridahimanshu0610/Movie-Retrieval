"""
Dense vector retriever backed by ChromaDB.

Accepts a list of field_type values for the WHERE filter so the same
retriever can be used for both single-field and multi-field queries
without separate collection scans.
"""

from __future__ import annotations

import chromadb

from video_config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, TOP_K
from video_indexer.embedder import Embedder

# Type alias for clarity
RetrievalResult = dict  # see _format() for the shape


class ChromaRetriever:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        field_types: list[str],
        top_k: int = TOP_K,
    ) -> list[RetrievalResult]:
        """
        Query ChromaDB filtered to the given field_type(s).

        Args:
            query:       The (optionally rewritten) search string.
            field_types: One or more field_type labels to restrict search to.
            top_k:       Maximum results to return.

        Returns:
            List of result dicts sorted by cosine similarity (descending).
            Each dict has the shape defined in _format_result().
        """
        if not field_types:
            raise ValueError("field_types must contain at least one element.")

        query_embedding = self.embedder.embed_one(query)
        where = _build_where_filter(field_types)

        # Guard: ChromaDB raises if n_results > collection size
        safe_k = min(top_k, self.collection.count())
        if safe_k == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=safe_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return [
            _format_result(
                doc_id=results["ids"][0][i],
                doc_text=results["documents"][0][i],
                meta=results["metadatas"][0][i],
                distance=results["distances"][0][i],
            )
            for i in range(len(results["ids"][0]))
        ]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_where_filter(field_types: list[str]) -> dict:
    """
    Build a ChromaDB WHERE clause.

    Single value → simple equality.
    Multiple values → $in operator.
    """
    if len(field_types) == 1:
        return {"field_type": {"$eq": field_types[0]}}
    return {"field_type": {"$in": field_types}}


def _format_result(
    doc_id: str,
    doc_text: str,
    meta: dict,
    distance: float,
) -> RetrievalResult:
    """
    Convert a raw ChromaDB result row into a clean result dict.

    Score: ChromaDB returns cosine *distance* (0 = identical, 2 = opposite).
    We convert to similarity in [0, 1] as:  similarity = 1 - distance / 2
    """
    return {
        "doc_id":       doc_id,
        "clip_id":      meta.get("clip_id", ""),
        "movie":        meta.get("movie", ""),
        "field_type":   meta.get("field_type", ""),
        "score":        round(1.0 - distance / 2.0, 4),
        "content":      doc_text,
        "youtube_link": meta.get("youtube_link", ""),
        "timestamp":    meta.get("timestamp", ""),
        "description":  meta.get("description", ""),
    }