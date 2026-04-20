"""
Thin wrapper around sentence-transformers for producing dense embeddings.
All embeddings are L2-normalised so cosine similarity == dot product.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from video_config import EMBEDDING_MODEL


class Embedder:
    """
    Singleton-style embedder.  Instantiate once and reuse across indexer
    and retriever to avoid loading the model multiple times.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        print(f"[Embedder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    # ── Public API ─────────────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors (L2-normalised).
        """
        return (
            self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            .tolist()
        )

    def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for a single string."""
        return self.embed([text])[0]