"""
query.py — interactive CLI for the movie scene retrieval system.

Modes
-----
Interactive REPL (default):
    python query.py --model /path/to/model.gguf

Single query (non-interactive):
    python query.py --model /path/to/model.gguf --query "scene with giant waves"

Skip LLM (keyword router only):
    python query.py --no-llm --query "astronauts in white suits near water"

GPU acceleration:
    python query.py --model /path/to/model.gguf --gpu-layers 32
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run directly.
sys.path.insert(0, str(Path(__file__).parent))

from config import LLM_MODEL_PATH, LLM_N_GPU_LAYERS, TOP_K
from pipeline import RetrievalPipeline


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the movie scene retrieval system."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=LLM_MODEL_PATH,
        help="Path to the .gguf model file.",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=LLM_N_GPU_LAYERS,
        dest="gpu_layers",
        help="Number of model layers to offload to GPU (default: 0 = CPU).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        dest="no_llm",
        help="Disable LLM router; use keyword fallback instead.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single query and exit (non-interactive mode).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        dest="top_k",
        help=f"Number of results to return (default: {TOP_K}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Print results as raw JSON instead of formatted text.",
    )
    return parser.parse_args()


# ── Formatting ─────────────────────────────────────────────────────────────────

def format_results(response: dict) -> str:
    lines: list[str] = []
    intent = response["intent"]

    lines.append("\n" + "═" * 60)
    lines.append(f"  Query : {response['query']}")
    lines.append(f"  Rewritten : {intent['rewritten_query']}")
    lines.append(
        f"  Routing   : primary={intent['primary_field']}  "
        f"secondary={intent['secondary_fields']}  "
    )
    lines.append("═" * 60)

    results = response["results"]
    if not results:
        lines.append("  No results found.")
        return "\n".join(lines)

    for i, r in enumerate(results, start=1):
        lines.append(f"\n  #{i}  [{r['movie']}]  clip: {r['clip_id']}")
        lines.append(f"       {r.get('description', '')}")
        if "llm_rerank_rank" in r:
            lines.append(
                f"       LLM rank       : {r['llm_rerank_rank']} "
                f"(retrieval rank {r.get('retrieval_rank', '?')})"
            )
        lines.append(
            f"       Matched fields : {', '.join(r.get('matched_fields', [r.get('field_type','?')]))}"
        )
        lines.append(f"       RRF score      : {r.get('rrf_score', r.get('score', '?'))}")
        lines.append(f"       Timestamp      : {r.get('timestamp', 'N/A')}")
        if r.get("youtube_link"):
            lines.append(f"       YouTube        : {r['youtube_link']}")
        # Show a short content snippet
        content = r.get("content", "")
        snippet = content[:200].replace("\n", " ")
        if len(content) > 200:
            snippet += " …"
        lines.append(f"       Snippet        : {snippet}")

    lines.append("")
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    use_llm = not args.no_llm
    pipeline = RetrievalPipeline(
        llm_model_path=args.model,
        n_gpu_layers=args.gpu_layers,
        use_llm=use_llm,
    )

    def run_query(q: str) -> None:
        response = pipeline.query(q, top_k=args.top_k)
        if args.output_json:
            print(json.dumps(response, indent=2, ensure_ascii=False))
        else:
            print(format_results(response))

    if args.query:
        # Non-interactive single query
        run_query(args.query)
    else:
        # Interactive REPL
        print("\nMovie Scene Retrieval System")
        print("Type a scene description and press Enter.  'quit' or Ctrl-C to exit.\n")
        while True:
            try:
                user_input = input("Search > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Exiting.")
                break

            run_query(user_input)


if __name__ == "__main__":
    main()
