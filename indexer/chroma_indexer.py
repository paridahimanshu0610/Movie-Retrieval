"""
Indexes movie clip metadata into ChromaDB at field-level granularity.

Strategy
--------
One ChromaDB document is created per (clip × field) pair — e.g.:
    Interstellar__gigantic_wave__visual
    Interstellar__gigantic_wave__emotional
    ...

Each document carries metadata so downstream WHERE filters can scope
retrieval to a specific field type without needing separate collections.

Movie-level documents (plot, themes, tone) are also stored with
clip_id="" to support movie-first retrieval.
"""

from __future__ import annotations

import chromadb

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CLIP_FIELDS,
    MOVIE_FIELDS,
)
from indexer.embedder import Embedder

# How many documents to upsert per ChromaDB call.
_BATCH_SIZE = 64


class ChromaIndexer:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def index(self, movie_data: dict) -> None:
        """
        Build and upsert all documents from the full movie data dict.
        Safe to re-run (upsert is idempotent by document id).
        """
        ids, documents, metadatas = [], [], []

        for movie_title, movie in movie_data.items():
            self._collect_movie_docs(movie_title, movie, ids, documents, metadatas)
            for clip_id, clip in movie.get("clips", {}).items():
                self._collect_clip_docs(
                    movie_title, clip_id, clip, ids, documents, metadatas
                )
        
        # print("ids:", ids)
        # print("documents:", documents)
        # print("metadatas:", metadatas)
        self._upsert_batched(ids, documents, metadatas)
        print(f"[ChromaIndexer] Upserted {len(documents)} documents.")

    def count(self) -> int:
        return self.collection.count()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _collect_movie_docs(
        self,
        movie_title: str,
        movie: dict,
        ids: list,
        documents: list,
        metadatas: list,
    ) -> None:
        for field_key, data_key in MOVIE_FIELDS.items():
            value = movie.get(data_key)
            if not value:
                continue
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)

            ids.append(f"{movie_title}__movie__{field_key}")
            documents.append(value)
            metadatas.append(
                _make_metadata(
                    movie=movie_title,
                    field_type=field_key,
                    clip_id="",
                    youtube_link="",
                    timestamp="",
                    description=f"{movie_title} – {field_key}",
                )
            )

    def _collect_clip_docs(
        self,
        movie_title: str,
        clip_id: str,
        clip: dict,
        ids: list,
        documents: list,
        metadatas: list,
    ) -> None:
        for field_key, data_key in CLIP_FIELDS.items():
            value = clip.get(data_key)
            if not value or not str(value).strip():
                continue

            ids.append(f"{movie_title}__{clip_id}__{field_key}")
            documents.append(str(value))
            metadatas.append(
                _make_metadata(
                    movie=movie_title,
                    field_type=field_key,
                    clip_id=clip_id,
                    youtube_link=clip.get("youtube_link", ""),
                    timestamp=clip.get("timestamp", ""),
                    description=clip.get("description", ""),
                )
            )

    def _upsert_batched(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        for start in range(0, len(ids), _BATCH_SIZE):
            end = start + _BATCH_SIZE
            batch_docs = documents[start:end]
            batch_embeddings = self.embedder.embed(batch_docs)
            self.collection.upsert(
                ids=ids[start:end],
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=metadatas[start:end],
            )
            print(
                f"[ChromaIndexer]  upserted batch {start}–{min(end, len(ids))} "
                f"/ {len(ids)}"
            )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_metadata(
    *,
    movie: str,
    field_type: str,
    clip_id: str,
    youtube_link: str,
    timestamp: str,
    description: str,
) -> dict:
    """
    ChromaDB metadata values must be str | int | float | bool.
    All fields are coerced to str here.
    """
    return {
        "movie": movie,
        "field_type": field_type,
        "clip_id": clip_id,
        "youtube_link": youtube_link,
        "timestamp": timestamp,
        "description": description,
    }