"""
ingest.py — one-time indexing script.

Run this BEFORE query.py to populate ChromaDB and the BM25 pickle.

Usage
-----
    python ingest.py
    python ingest.py --data path/to/final_clip_data.json
    python ingest.py --reset   # wipe and re-index from scratch
"""

import argparse
import shutil
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run directly.
sys.path.insert(0, str(Path(__file__).parent))

from config import BM25_INDEX_PATH, CHROMA_PERSIST_DIR, DATA_PATH
from data_loader import load_clip_data
from indexer.bm25_indexer import BM25Indexer
from indexer.chroma_indexer import ChromaIndexer
from indexer.embedder import Embedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index movie clip data.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Path to final_clip_data.json (default: outputs/final_clip_data.json)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing ChromaDB and BM25 index before re-indexing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.reset:
        print("[Ingest] --reset flag set: wiping existing indexes …")
        if CHROMA_PERSIST_DIR.exists():
            shutil.rmtree(CHROMA_PERSIST_DIR)
            print(f"[Ingest] Removed {CHROMA_PERSIST_DIR}")
        if BM25_INDEX_PATH.exists():
            BM25_INDEX_PATH.unlink()
            print(f"[Ingest] Removed {BM25_INDEX_PATH}")

    # ── Load data ──────────────────────────────────────────────────────────────
    movie_data = load_clip_data(args.data)

    # ── Embed + index into ChromaDB ────────────────────────────────────────────
    print("\n[Ingest] Building ChromaDB index …")
    embedder = Embedder()
    chroma_indexer = ChromaIndexer(embedder)
    chroma_indexer.index(movie_data)
    print(f"[Ingest] ChromaDB total documents: {chroma_indexer.count()}")

    # ── Build BM25 over dialogue ───────────────────────────────────────────────
    print("\n[Ingest] Building BM25 dialogue index …")
    bm25_indexer = BM25Indexer()
    bm25_indexer.build(movie_data)
    bm25_indexer.save()

    print("\n[Ingest] Done.  You can now run query.py.")


if __name__ == "__main__":
    main()
