"""
Screenplay retrieval evaluation script.

Runs a set of test queries directly through the retrieval pipeline and computes:
  MRR, Hit@K, Precision@K, Recall@K, NDCG@K, Intent Accuracy

Usage:
  python evaluate.py
  python evaluate.py --k 5 --verbose
  python evaluate.py --mode bm25 --output eval_results.json
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.resolve()
RETRIEVAL_DIR = ROOT / "retrieval"
INDEX_DIR    = ROOT / "indices"
ENV_PATH     = ROOT / ".env"

sys.path.insert(0, str(RETRIEVAL_DIR))

import retriever as ret
from intent_classifier import classify_query
from reranker import ScreenplayReranker

ret.INDEX_DIR = INDEX_DIR

# ── One-time setup ────────────────────────────────────────────────────────────
ret.preload_indices()

_registry: dict = json.loads((INDEX_DIR / "registry.json").read_text(encoding="utf-8"))
_reranker = ScreenplayReranker(registry=_registry)
_TITLE_MAP: dict = {mid: info.get("title", f"Movie {mid}") for mid, info in _registry.items()}


def _fuzzy_title_to_ids(titles: list[str]) -> set:
    ids = set()
    for title in titles:
        tl = title.lower().strip()
        for mid, info in _registry.items():
            rt = info.get("title", "").lower().strip()
            if tl == rt or tl in rt or rt in tl:
                ids.add(mid)
    return ids

# ── Test queries ──────────────────────────────────────────────────────────────
# ground_truth: list of acceptable movie titles (lowercase, partial match ok)
# expected_intent: the query_type you expect the router to assign (for intent acc)
# Set expected_intent to None to skip that query from intent accuracy calculation.

QUERIES = [
    # --- Dialogue ---
    {
        "query": '"Get busy living, or get busy dying"',
        "ground_truth": ["shawshank"],
        "expected_intent": "dialogue",
    },
    {
        "query": '"Life finds a way"',
        "ground_truth": ["jurassic park"],
        "expected_intent": "dialogue",
    },
    {
        "query": '"You shall not pass"',
        "ground_truth": ["lord of the rings", "fellowship"],
        "expected_intent": "dialogue",
    },

    # --- Simple Scene ---
    
    {
        "query": 'A character quotes a Bible verse about vengeance before executing two men in an apartment',
        "ground_truth": ["pulp fiction"],
        "expected_intent": "detailed_scene",
    },

    {
        "query": "A getaway driver listens to music through earbuds while his crew robs a coffee shop",
        "ground_truth": ["baby driver"],
        "expected_intent": "simple_scene",
    },
    {
        "query": "An old widower ties thousands of balloons to his house and floats away with a young stowaway toward a waterfall in South America",
        "ground_truth": ["up"],
        "expected_intent": "simple_scene",
    },
    {
        "query": "A woman trapped in a coffin underground breaks free using a technique from her martial arts training",
        "ground_truth": ["kill bill"],
        "expected_intent": "detailed_scene",
    },
    {
        "query": "A superhero flies through New York City carrying a nuclear weapon toward a portal in the sky",
        "ground_truth": ["avengers"],
        "expected_intent": "simple_scene",
    },
    {
        "query": "A man in a suit of armor escapes captivity by flying through the desert",
        "ground_truth": ["iron man"],
        "expected_intent": "simple_scene",
    },

    # --- Plot Level ---
    {
        "query": "A wrongfully imprisoned banker befriends a lifer and escapes through a tunnel he dug over decades",
        "ground_truth": ["shawshank"],
        "expected_intent": "plot_level",
    },
    {
        "query": "A superhero team assembles for the first time to stop an alien army invading New York",
        "ground_truth": ["avengers"],
        "expected_intent": "plot_level",
    },
    {
        "query": "An engineer builds a robotic suit of armor after being captured by terrorists in a cave",
        "ground_truth": ["iron man"],
        "expected_intent": "plot_level",
    },
    {
        "query": "Astronauts travel through a wormhole to find a habitable planet as Earth becomes uninhabitable",
        "ground_truth": ["interstellar"],
        "expected_intent": "plot_level",
    },

    # --- Character Specific ---
    {
        "query": "A lawyer uses her fashion sense and unexpected intelligence to win a murder case",
        "ground_truth": ["legally blonde"],
        "expected_intent": "plot_level",
    },
    {
        "query": "A bounty hunter and a freed slave team up to rescue the slave's wife from a plantation",
        "ground_truth": ["django"],
        "expected_intent": "plot_level",
    },
    {
        "query": "An immortal mercenary who cannot die constantly breaks the fourth wall and makes jokes",
        "ground_truth": ["deadpool"],
        "expected_intent": "plot_level",
    },
    {
        "query": "A jeweler obsessed with a rare opal makes increasingly reckless bets to pay off his debts",
        "ground_truth": ["uncut gems"],
        "expected_intent": "plot_level",
    },

    # --- Filtered ---
    {
        "query": "A horror film from 2019 set during a midsummer festival in Sweden",
        "ground_truth": ["midsommar"],
        "expected_intent": "filtered",
    },
    {
        "query": "A Tarantino film from 1992 where criminals argue about tipping before robbing a diner",
        "ground_truth": ["reservoir dogs"],
        "expected_intent": "filtered",
    },
    {
        "query": "An animated Pixar film about toys who fear being donated to a daycare",
        "ground_truth": ["toy story 3"],
        "expected_intent": "filtered",
    },
    {
        "query": "A 2014 Christopher Nolan film about space travel and time dilation near a black hole",
        "ground_truth": ["interstellar"],
        "expected_intent": "filtered",
    },
    {
        "query": "A superhero film featuring both Wolverine and Deadpool fighting alongside each other",
        "ground_truth": ["deadpool"],
        "expected_intent": "filtered",
    },

    # --- Multi-aspect ---
    {
        "query": "A dark scene inside a flooded football stadium where a villain addresses a terrified crowd, with a threatening and anarchic tone",
        "ground_truth": ["dark knight rises"],
        "expected_intent": "complex_scene",
    },
    {
        "query": "A group of misfit student monsters must win a university scaring competition or face expulsion",
        "ground_truth": ["monsters university"],
        "expected_intent": "plot_level",
    },

    # --- Negation ---
    {
        "query": "A movie about a family of superheroes who hide their powers and live in the suburbs, not The Avengers",
        "ground_truth": ["incredibles"],
        "expected_intent": "negation",
    },
    {
        "query": "A slow-burn horror film set in an isolated location where characters gradually turn on each other, but not Midsommar",
        "ground_truth": ["lighthouse", "us", "nope"],
        "expected_intent": "negation",
    },

    # --- Similarity ---
    {
        "query": "A movie similar to Pulp Fiction with interconnected crime stories told out of chronological order",
        "ground_truth": ["reservoir dogs", "jackie brown", "true romance"],
        "expected_intent": "similarity",
    },

    # --- Additional ---
    {
        "query": "Two kids driving a flying car being chased by a train",
        "ground_truth": ["harry potter", "chamber of secrets"],
        "expected_intent": "simple_scene",
    },
    {
        "query": "A getaway driver subtly signals a woman to leave the area while his accomplices rob a place",
        "ground_truth": ["baby driver"],
        "expected_intent": "detailed_scene",
    },
    {
        "query": "A movie where Woody Harrelson plays a bounty hunter chasing a psychopath to recover money",
        "ground_truth": ["no country for old men"],
        "expected_intent": "plot_level",
    },
]


# ── Metrics ───────────────────────────────────────────────────────────────────

def _is_match(title: str, ground_truth: list[str]) -> bool:
    title_lower = title.lower()
    return any(gt in title_lower for gt in ground_truth)


def reciprocal_rank(titles: list[str], ground_truth: list[str]) -> float:
    for rank, title in enumerate(titles, 1):
        if _is_match(title, ground_truth):
            return 1.0 / rank
    return 0.0


def hit_at_k(titles: list[str], ground_truth: list[str]) -> int:
    return int(any(_is_match(t, ground_truth) for t in titles))


def precision_at_k(titles: list[str], ground_truth: list[str]) -> float:
    hits = sum(_is_match(t, ground_truth) for t in titles)
    return hits / len(titles) if titles else 0.0


def recall_at_k(titles: list[str], ground_truth: list[str]) -> float:
    # With one correct answer per query, recall@k == hit@k
    return float(hit_at_k(titles, ground_truth))


def ndcg_at_k(titles: list[str], ground_truth: list[str]) -> float:
    dcg = sum(
        _is_match(title, ground_truth) / math.log2(rank + 1)
        for rank, title in enumerate(titles, 1)
    )
    idcg = 1.0  # ideal: correct answer at rank 1 → 1/log2(2) = 1
    return dcg / idcg if idcg > 0 else 0.0


# ── Runner ────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query:           str
    ground_truth:    list[str]
    expected_intent: str | None
    returned_titles: list[str]
    returned_intent: str
    rr:   float = 0.0
    hit:  int   = 0
    prec: float = 0.0
    rec:  float = 0.0
    ndcg: float = 0.0
    intent_correct: bool | None = None
    latency_s: float = 0.0
    error: str = ""


def run_query(query: str, k: int, mode: str) -> tuple[list[str], str, float]:
    t0 = time.time()

    plan = classify_query(query, env_path=ENV_PATH)
    plan.pop("_raw_response", None)
    plan.pop("_mode", None)

    f          = plan.get("filters", {})
    year_range = (f.get("year_min"), f.get("year_max")) if (f.get("year_min") or f.get("year_max")) else None
    exclude_ids = _fuzzy_title_to_ids(plan.get("exclude_titles", []))
    raw_sqs     = plan.get("sub_queries")

    raw_results, _ = ret.retrieve(
        query        = plan.get("rewrite") or query,
        intents      = plan.get("intents", ["full"]),
        mode         = mode,
        top_k        = k,
        genre_filter = f.get("genre"),
        year_range   = year_range,
        exclude_ids  = exclude_ids,
        sub_queries  = raw_sqs,
    )

    raw_results = _reranker.rerank(query, raw_results, k)

    titles  = [_TITLE_MAP.get(r["movie_id"], f"Movie {r['movie_id']}") for r in raw_results]
    intent  = plan.get("query_type", "unknown")
    latency = time.time() - t0
    return titles, intent, latency


def evaluate(k: int, mode: str, verbose: bool) -> list[QueryResult]:
    results = []
    for i, q in enumerate(QUERIES, 1):
        print(f"[{i:02d}/{len(QUERIES)}] {q['query'][:70]}...")
        result = QueryResult(
            query           = q["query"],
            ground_truth    = q["ground_truth"],
            expected_intent = q.get("expected_intent"),
            returned_titles = [],
            returned_intent = "",
        )
        try:
            titles, intent, latency = run_query(q["query"], k, mode)
            result.returned_titles  = titles
            result.returned_intent  = intent
            result.latency_s        = latency
            result.rr               = reciprocal_rank(titles, q["ground_truth"])
            result.hit              = hit_at_k(titles, q["ground_truth"])
            result.prec             = precision_at_k(titles, q["ground_truth"])
            result.rec              = recall_at_k(titles, q["ground_truth"])
            result.ndcg             = ndcg_at_k(titles, q["ground_truth"])
            if q.get("expected_intent"):
                result.intent_correct = (intent == q["expected_intent"])
            if verbose:
                print(f"         intent={intent}  results={titles}")
                print(f"         RR={result.rr:.3f}  hit={result.hit}  NDCG={result.ndcg:.3f}  latency={latency:.1f}s")
        except Exception as e:
            result.error = str(e)
            print(f"         ERROR: {e}")
        results.append(result)
    return results


def print_report(results: list[QueryResult], k: int) -> None:
    valid   = [r for r in results if not r.error]
    n       = len(valid)
    if n == 0:
        print("No valid results.")
        return

    mrr     = sum(r.rr   for r in valid) / n
    hit     = sum(r.hit  for r in valid) / n
    prec    = sum(r.prec for r in valid) / n
    rec     = sum(r.rec  for r in valid) / n
    ndcg    = sum(r.ndcg for r in valid) / n
    avg_lat = sum(r.latency_s for r in valid) / n

    intent_judged = [r for r in valid if r.intent_correct is not None]
    intent_acc    = sum(r.intent_correct for r in intent_judged) / len(intent_judged) if intent_judged else float("nan")

    print("\n" + "=" * 60)
    print(f"  Evaluation results  (K={k}, N={n} queries)")
    print("=" * 60)
    print(f"  MRR          : {mrr:.3f}")
    print(f"  Hit@{k}        : {hit:.3f}")
    print(f"  Precision@{k}  : {prec:.3f}  (ceiling = {1/k:.3f})")
    print(f"  Recall@{k}     : {rec:.3f}")
    print(f"  NDCG@{k}       : {ndcg:.3f}")
    print(f"  Intent Acc   : {intent_acc:.3f}  ({sum(r.intent_correct for r in intent_judged)}/{len(intent_judged)})")
    print(f"  Avg latency  : {avg_lat:.2f}s")
    print("=" * 60)

    # Per-query breakdown
    print(f"\n{'#':<4} {'Intent':20} {'RR':>5} {'Hit':>4} {'NDCG':>6}  Query")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        if r.error:
            print(f"{i:<4} {'ERROR':20}                   {r.query[:45]}")
        else:
            tick = "" if r.intent_correct is None else ("OK" if r.intent_correct else "X")
            print(f"{i:<4} {r.returned_intent:20} {r.rr:>5.3f} {r.hit:>4} {r.ndcg:>6.3f}  [{tick}] {r.query[:45]}")

    # Failed queries
    failed = [r for r in valid if r.hit == 0]
    if failed:
        print(f"\nMissed ({len(failed)}):")
        for r in failed:
            print(f"  - {r.query[:70]}")
            print(f"    expected: {r.ground_truth}")
            print(f"    got:      {r.returned_titles}")


def save_json(results: list[QueryResult], path: str) -> None:
    out = []
    for r in results:
        out.append({
            "query":           r.query,
            "ground_truth":    r.ground_truth,
            "expected_intent": r.expected_intent,
            "returned_titles": r.returned_titles,
            "returned_intent": r.returned_intent,
            "intent_correct":  r.intent_correct,
            "rr":              round(r.rr,   4),
            "hit":             r.hit,
            "precision":       round(r.prec, 4),
            "recall":          round(r.rec,  4),
            "ndcg":            round(r.ndcg, 4),
            "latency_s":       round(r.latency_s, 2),
            "error":           r.error,
        })
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",       type=int, default=5,                        help="Top-K for evaluation (default 5)")
    parser.add_argument("--mode",    type=str, default="hybrid",                  choices=["faiss", "bm25", "hybrid"], help="Retrieval mode (default hybrid)")
    parser.add_argument("--verbose", action="store_true",                         help="Print per-query results and returned titles")
    parser.add_argument("--output",  type=str, default="eval_results.json",       help="JSON output path")
    args = parser.parse_args()

    print(f"Evaluating  (K={args.k}, mode={args.mode}, {len(QUERIES)} queries)\n")

    results = evaluate(k=args.k, mode=args.mode, verbose=args.verbose)
    print_report(results, k=args.k)
    save_json(results, args.output)
