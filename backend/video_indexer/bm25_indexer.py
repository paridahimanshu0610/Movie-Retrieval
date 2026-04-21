"""
BM25 (Okapi BM25) index built exclusively over dialogue/transcript text.

Why BM25 for dialogue?
----------------------
When a user types an exact or near-exact quote from a scene (e.g. "those
aren't mountains they're waves"), dense vector search may not rank it
#1 because the embedding space smooths over lexical specifics.  BM25 is
a term-frequency model that naturally rewards exact token matches,
making it the right tool for verbatim dialogue queries.

The index is persisted as a pickle file alongside the ChromaDB store so
it can be loaded without re-ingesting on every query.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from video_config import BM25_INDEX_PATH


# ── Tokenisation ───────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Lowercase, strip punctuation, split on whitespace.
    Keeps stopwords intentionally — BM25 handles term saturation naturally
    and removing stopwords can hurt recall on short dialogue queries.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # punctuation → space
    return text.split()


# ── Indexer ────────────────────────────────────────────────────────────────────

class BM25Indexer:
    """
    Builds and persists a BM25 index over all clip dialogue transcripts.

    Attributes
    ----------
    corpus_meta : list[dict]
        One entry per indexed clip with keys:
        movie, clip_id, youtube_link, timestamp, description.
    corpus_texts : list[str]
        Raw (un-tokenised) transcript text — kept for snippet display.
    bm25 : BM25Okapi | None
        The fitted BM25 model.
    """

    def __init__(self) -> None:
        self.corpus_meta: list[dict] = []
        self.corpus_texts: list[str] = []
        self.bm25: BM25Okapi | None = None

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, movie_data: dict) -> None:
        """
        Iterate over all clips and populate the BM25 index from their
        concatenated_transcript field.  Clips with no transcript are skipped.
        """
        for movie_title, movie in movie_data.items():
            for clip_id, clip in movie.get("clips", {}).items():
                transcript = (clip.get("concatenated_transcript") or "").strip()
                if not transcript:
                    continue

                self.corpus_meta.append(
                    {
                        "movie": movie_title,
                        "clip_id": clip_id,
                        "youtube_link": clip.get("youtube_link", ""),
                        "timestamp": clip.get("timestamp", ""),
                        "description": clip.get("description", ""),
                    }
                )
                self.corpus_texts.append(transcript)

        tokenized_corpus = [tokenize(t) for t in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(
            f"[BM25Indexer] Built index over {len(self.corpus_texts)} dialogue entries."
        )

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Path = BM25_INDEX_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "corpus_meta": self.corpus_meta,
                    "corpus_texts": self.corpus_texts,
                    "bm25": self.bm25,
                },
                f,
            )
        print(f"[BM25Indexer] Saved to {path}")

    @classmethod
    def load(cls, path: Path = BM25_INDEX_PATH) -> "BM25Indexer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {path}. Run ingest.py first."
            )
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls()
        instance.corpus_meta = data["corpus_meta"]
        instance.corpus_texts = data["corpus_texts"]
        instance.bm25 = data["bm25"]
        print(f"[BM25Indexer] Loaded index ({len(instance.corpus_texts)} entries).")
        return instance