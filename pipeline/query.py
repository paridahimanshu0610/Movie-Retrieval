"""
query.py — Interactive query interface for the scene retrieval system.

Usage
─────
    # Single query
    python pipeline/query.py "a man in a spacesuit floating alone in orbit"

    # Interactive mode (no args)
    python pipeline/query.py
"""

import sys

import config
from embedders.text_embedder import TextEmbedder
from core.indexer import DualIndex, Retriever


def run_query(query: str, top_k: int = 5) -> None:
    # Load index
    index = DualIndex()
    index.load(config.INDEX_DIR)

    # Load only the lightweight text embedder (no GPU-heavy models needed at query time)
    text_embedder = TextEmbedder()
    retriever = Retriever(index, text_embedder)

    # Search
    results = retriever.search(query, top_k=top_k)

    print(f"\n{'═'*60}")
    print(f"  Query : \"{query}\"")
    print(f"  Top-{top_k} results  (RRF + dual-index)")
    print(f"{'═'*60}")

    for rank, r in enumerate(results, start=1):
        print(f"\n  #{rank}  [{r['movie']}]  {r['clip_id']}")
        print(f"      RRF score   : {r['rrf_score']}")
        print(f"      Visual cos  : {r['visual_score']}")
        print(f"      Text cos    : {r['text_score']}")
        print(f"      Caption     : {r['caption'][:120]} …")
        if r.get("transcript"):
            print(f"      Transcript  : {r['transcript'][:100]} …")
        print(f"      Path        : {r['path']}")

    print(f"\n{'═'*60}\n")


def interactive_mode() -> None:
    print("\nScene Retrieval — Interactive Mode  (Ctrl-C to quit)\n")
    while True:
        try:
            query = input("🎬 Describe a scene: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not query:
            continue
        run_query(query)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_query(" ".join(sys.argv[1:]))
    else:
        interactive_mode()
