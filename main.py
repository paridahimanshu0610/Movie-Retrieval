"""
main.py — Run the full screenplay retrieval pipeline, or query the indices.

Pipeline steps (in order):
  1. sync       — Register new screenplay files in manifest.json
  2. fetch      — Fetch movie metadata from TMDB
  3. convert    — Convert PDF/HTML/TXT files to normalized plain-text
  4. parse      — Parse plain-text into structured scene JSON
  5. reconcile  — Map screenplay character names to TMDB cast entries
  6. index      — Build FAISS, BM25, and structured retrieval indices

Usage:
  python main.py                                    # run all pipeline steps
  python main.py --from convert                     # start from a specific step
  python main.py --only index                       # run a single step
  python main.py --dry-run                          # preview without writing

  python main.py --query "car chase with loud music"
  python main.py --query "what are you listening to" --intent dialogue
  python main.py --query "two brothers run a juke joint" --intent plot
  python main.py --query "villain monologue before a fight" --intent dialogue,scene
  python main.py --query "..." --mode bm25 --top-k 10
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
PARSER_DIR = ROOT / "screenplay_parser"
RETRIEVAL_DIR = ROOT / "retrieval"

STEPS = [
    {
        "name": "sync",
        "label": "Sync manifest with screenplays directory",
        "script": PARSER_DIR / "manifest_sync.py",
        "args": [
            "--screenplays-dir", str(ROOT / "screenplays"),
            "--manifest",        str(ROOT / "screenplays" / "manifest.json"),
        ],
        "supports_dry_run": True,
    },
    {
        "name": "fetch",
        "label": "Fetch TMDB metadata",
        "script": PARSER_DIR / "tmdb_fetch.py",
        "args": [
            "--manifest",    str(ROOT / "screenplays" / "manifest.json"),
            "--output-dir",  str(ROOT / "output"),
            "--env",         str(ROOT / ".env"),
        ],
        "supports_dry_run": True,
    },
    {
        "name": "convert",
        "label": "Convert screenplay files to plain-text",
        "script": PARSER_DIR / "convert.py",
        "args": [
            "--input-dir",   str(ROOT / "screenplays"),
            "--manifest",    str(ROOT / "screenplays" / "manifest.json"),
            "--output-dir",  str(ROOT / "screenplays" / "converted"),
        ],
        "supports_dry_run": False,
    },
    {
        "name": "parse",
        "label": "Parse plain-text into scene JSON",
        "script": PARSER_DIR / "batch.py",
        "args": [
            "--converted-dir", str(ROOT / "screenplays" / "converted"),
            "--manifest",      str(ROOT / "screenplays" / "manifest.json"),
            "--output-dir",    str(ROOT / "output"),
        ],
        "supports_dry_run": False,
    },
    {
        "name": "reconcile",
        "label": "Reconcile screenplay character names with TMDB cast",
        "script": PARSER_DIR / "character_reconcile.py",
        "args": [
            "--manifest",   str(ROOT / "screenplays" / "manifest.json"),
            "--scenes-dir", str(ROOT / "output"),
            "--output-dir", str(ROOT / "output"),
        ],
        "supports_dry_run": False,
    },
    {
        "name": "index",
        "label": "Build retrieval indices",
        "script": RETRIEVAL_DIR / "index_builder.py",
        "args": [
            "--manifest",   str(ROOT / "screenplays" / "manifest.json"),
            "--input-dir",  str(ROOT / "output"),
            "--index-dir",  str(ROOT / "indices"),
        ],
        "supports_dry_run": False,
    },
]

STEP_NAMES = [s["name"] for s in STEPS]


def run_step(step: dict, dry_run: bool = False) -> bool:
    print(f"\n{'='*60}")
    print(f"  {step['label']}")
    print(f"{'='*60}")

    cmd = [sys.executable, str(step["script"])] + step["args"]
    if dry_run and step["supports_dry_run"]:
        cmd.append("--dry-run")

    result = subprocess.run(cmd, cwd=str(step["script"].parent))
    if result.returncode != 0:
        print(f"\nStep '{step['name']}' failed (exit code {result.returncode}).", file=sys.stderr)
        return False
    return True


def _fuzzy_title_to_ids(titles: list, registry: dict) -> set:
    """Fuzzy movie title -> movie_id lookup (case-insensitive substring)."""
    ids: set = set()
    for title in titles:
        tl = title.lower().strip()
        for mid, info in registry.items():
            rt = info.get("title", "").lower().strip()
            if tl == rt or tl in rt or rt in tl:
                ids.add(mid)
    return ids


def _print_results(
    query: str,
    plan: dict,
    results: list,
    entity_names: list,
    registry: dict,
) -> None:
    width = 72
    filters = plan.get("filters", {})
    print(plan)
    print()
    print("=" * width)
    print(f"  Query  : {query}")
    print(f"  Type   : {plan['query_type']}   Mode : {plan.get('_mode', 'hybrid')}")
    print(f"  Intent : {', '.join(plan['intents'])}")
    if entity_names:
        print(f"  Entities: {', '.join(entity_names)}")
    if plan.get("reference_title"):
        print(f"  Ref    : {plan['reference_title']}")
    if plan.get("exclude_titles"):
        print(f"  Exclude: {', '.join(plan['exclude_titles'])}")
    if filters.get("genre"):
        print(f"  Genre  : {filters['genre']}")
    yr_min, yr_max = filters.get("year_min"), filters.get("year_max")
    if yr_min or yr_max:
        print(f"  Years  : {yr_min or '?'} – {yr_max or '?'}")
    if plan.get("rewrite"):
        print(f"  Rewrite: {plan['rewrite'][:width - 12]}")
    if plan.get("_raw_response"):
        print(f"  LLM raw: {plan['_raw_response'][:width - 11]}")
    print("=" * width)

    title_map = {mid: info["title"] for mid, info in registry.items()}

    for rank, r in enumerate(results, 1):
        mid   = r["movie_id"]
        title = title_map.get(mid, f"Movie {mid}")
        score = r["score"]

        print()
        print(f"  #{rank}  {title}  (id: {mid})  score: {score:.4f}")
        print(f"  {'-' * (width - 4)}")

        for s in r["top_scenes"]:
            print(f"    scene  : {s.get('scene_id', s.get('movie_id', ''))}")
            print(f"    heading: {s.get('heading', s.get('title', ''))}")
            snippet = s.get("text", s.get("overview", ""))[:100].replace("\n", " ")
            full    = s.get("text", s.get("overview", ""))
            print(f"    text   : {snippet}{'...' if len(full) > 100 else ''}")
            print()


def run_query(query: str, intent: str | None, mode: str, top_k: int) -> None:
    sys.path.insert(0, str(RETRIEVAL_DIR))
    try:
        import retriever as ret
        from intent_classifier import classify_query

        ret.INDEX_DIR = ROOT / "indices"

        # Build query plan
        if intent is not None:
            plan = {
                "query_type":      "manual",
                "intents":         [i.strip() for i in intent.split(",")],
                "filters":         {"genre": None, "year_min": None, "year_max": None},
                "exclude_titles":  [],
                "reference_title": None,
                "rewrite":         None,
            }
        else:
            plan = classify_query(query, env_path=ROOT / ".env")

        plan["_mode"] = mode  # carry mode for display

        # Load registry for title lookups and display
        reg_path = ROOT / "indices" / "registry.json"
        registry = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else {}

        # Resolve excluded titles to movie_ids
        exclude_ids = _fuzzy_title_to_ids(plan.get("exclude_titles", []), registry)

        # Year range
        f = plan.get("filters", {})
        year_range = (f.get("year_min"), f.get("year_max")) if (f.get("year_min") or f.get("year_max")) else None

        # Use rewrite as search text when the plan provides one
        search_query = plan.get("rewrite") or query

        results, entity_names = ret.retrieve(
            query        = search_query,
            intents      = plan["intents"],
            mode         = mode,
            top_k        = top_k,
            genre_filter = f.get("genre"),
            year_range   = year_range,
            exclude_ids  = exclude_ids,
        )

        _print_results(query, plan, results[:top_k], entity_names, registry)

    finally:
        sys.path.pop(0)


def main():
    ap = argparse.ArgumentParser(
        description="Run the screenplay retrieval pipeline, or query the indices."
    )

    # Query mode
    ap.add_argument("--query",   default=None, help="Natural-language query (activates query mode)")
    ap.add_argument("--intent",  default=None,
                    help="Comma-separated intents: scene,dialogue,character,full,plot "
                         "(default: auto-detected via LLM)")
    ap.add_argument("--mode",    default="hybrid", choices=["faiss", "bm25", "hybrid"])
    ap.add_argument("--top-k",   type=int, default=5)

    # Pipeline mode
    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--from", dest="from_step", metavar="STEP",
        choices=STEP_NAMES,
        help=f"Start pipeline from this step. Choices: {', '.join(STEP_NAMES)}",
    )
    group.add_argument(
        "--only", dest="only_step", metavar="STEP",
        choices=STEP_NAMES,
        help="Run only this pipeline step.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Pass --dry-run to steps that support it (sync, fetch).",
    )
    args = ap.parse_args()

    if args.query:
        run_query(args.query, args.intent, args.mode, args.top_k)
        return

    if args.only_step:
        steps_to_run = [s for s in STEPS if s["name"] == args.only_step]
    elif args.from_step:
        start = STEP_NAMES.index(args.from_step)
        steps_to_run = STEPS[start:]
    else:
        steps_to_run = STEPS

    print(f"Running {len(steps_to_run)} step(s): {', '.join(s['name'] for s in steps_to_run)}")

    for step in steps_to_run:
        ok = run_step(step, dry_run=args.dry_run)
        if not ok:
            sys.exit(1)

    print(f"\nPipeline complete.")


if __name__ == "__main__":
    main()
