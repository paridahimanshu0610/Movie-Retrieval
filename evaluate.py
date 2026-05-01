"""
evaluate.py — Evaluation harness for the movie scene retrieval system.

Metrics computed
----------------
  MRR@K      Mean Reciprocal Rank          — how high the first hit appears
  Hit Rate@K At least one hit in top-K     — coarse "does it work?" signal
  Precision@K Fraction of top-K that are relevant
  Recall@K   Fraction of relevant found in top-K
  NDCG@K     Rank-weighted relevance       — the richest single metric

Additionally:
  Intent Accuracy   Fraction of queries where the router's primary_field
                    matches the expected intent label in the test set.
                    (Only reported when expected_intent is provided.)

Usage
-----
  # No LLM required (keyword router + dense + BM25):
  python evaluate.py

  # Full LLM pipeline:
  python evaluate.py --use-llm --model /path/to/model.gguf

  # Custom query file and K:
  python evaluate.py --queries eval_queries.json --k 5

  # Save a detailed per-query JSON report:
  python evaluate.py --save-report results/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Allow running from the project root without installing the package.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from config import LLM_MODEL_PATH, LLM_N_GPU_LAYERS, TOP_K, EVAL_RESULTS_PATH
from pipeline import RetrievalPipeline

# ---------------------------------------------------------------------------
# Default path to ground-truth queries (relative to this file)
# ---------------------------------------------------------------------------
DEFAULT_QUERIES_PATH = Path(__file__).parent / "eval_queries.json"

# ===========================================================================
# Metric helpers
# ===========================================================================

def _is_relevant(result: dict, relevant_set: set[tuple[str, str]]) -> bool:
    """Return True if a result dict matches any (movie, clip_id) in the set."""
    return (result.get("movie", ""), result.get("clip_id", "")) in relevant_set


def reciprocal_rank(results: list[dict], relevant_set: set[tuple[str, str]]) -> float:
    """
    1 / rank of the first relevant result; 0 if none found.

    Args:
        results:      Ranked list of result dicts (index 0 = rank 1).
        relevant_set: Set of (movie, clip_id) tuples that count as relevant.

    Returns:
        Float in [0, 1].
    """
    for rank_0based, result in enumerate(results):
        if _is_relevant(result, relevant_set):
            return 1.0 / (rank_0based + 1)
    return 0.0


def precision_at_k(results: list[dict], relevant_set: set[tuple[str, str]], k: int) -> float:
    """
    Fraction of the top-k results that are relevant.

    Args:
        results:      Full ranked list.
        relevant_set: Relevant (movie, clip_id) pairs.
        k:            Cutoff.

    Returns:
        Float in [0, 1].
    """
    top = results[:k]
    if not top:
        return 0.0
    hits = sum(1 for r in top if _is_relevant(r, relevant_set))
    return hits / len(top)


def recall_at_k(results: list[dict], relevant_set: set[tuple[str, str]], k: int) -> float:
    """
    Fraction of all relevant items found in the top-k results.

    Args:
        results:      Full ranked list.
        relevant_set: Relevant (movie, clip_id) pairs.
        k:            Cutoff.

    Returns:
        Float in [0, 1].  Returns 0 when relevant_set is empty.
    """
    if not relevant_set:
        return 0.0
    top = results[:k]
    hits = sum(1 for r in top if _is_relevant(r, relevant_set))
    return hits / len(relevant_set)


def hit_rate_at_k(results: list[dict], relevant_set: set[tuple[str, str]], k: int) -> float:
    """
    Binary: 1.0 if any of the top-k results is relevant, else 0.0.

    This is the most forgiving metric — useful for small K values where
    even a single correct answer signals the system is working.

    Args:
        results:      Full ranked list.
        relevant_set: Relevant (movie, clip_id) pairs.
        k:            Cutoff.

    Returns:
        1.0 or 0.0.
    """
    return 1.0 if any(_is_relevant(r, relevant_set) for r in results[:k]) else 0.0


def ndcg_at_k(results: list[dict], relevant_set: set[tuple[str, str]], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at cutoff k.

    Binary relevance (1 if relevant, 0 otherwise).  The ideal DCG assumes
    all relevant items appear at the top of the list.

    NDCG@K = DCG@K / IDCG@K

    Args:
        results:      Full ranked list.
        relevant_set: Relevant (movie, clip_id) pairs.
        k:            Cutoff.

    Returns:
        Float in [0, 1].
    """
    def _dcg(hits: list[int]) -> float:
        return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(hits))

    top = results[:k]
    hits = [1 if _is_relevant(r, relevant_set) else 0 for r in top]

    dcg = _dcg(hits)

    # Ideal DCG: place all relevant items at the top, capped at k
    ideal_hits = [1] * min(len(relevant_set), k) + [0] * max(0, k - len(relevant_set))
    idcg = _dcg(ideal_hits[:k])

    return dcg / idcg if idcg > 0 else 0.0


# ===========================================================================
# Per-query evaluation
# ===========================================================================

def evaluate_query(
    pipeline: RetrievalPipeline,
    query_obj: dict,
    k: int,
) -> dict:
    """
    Run the pipeline on a single query and compute all metrics.

    Args:
        pipeline:   Initialised RetrievalPipeline.
        query_obj:  One entry from the eval_queries.json file.
        k:          Metric cutoff.

    Returns:
        Dict with keys: id, query, metrics, intent_correct, latency_s,
        expected_clips, retrieved_clips, raw_results.
    """
    query_text = query_obj["query"]
    relevant_clips = query_obj.get("relevant_clips", [])
    relevant_set = {(rc["movie"], rc["clip_id"]) for rc in relevant_clips}
    expected_intent = query_obj.get("expected_intent")

    t0 = time.perf_counter()
    response = pipeline.query(query_text, top_k=k)
    latency = time.perf_counter() - t0

    results = response.get("results", [])
    routed_intent = response.get("intent", {}).get("primary_field", "")

    intent_correct: bool | None = None
    if expected_intent:
        intent_correct = routed_intent == expected_intent

    metrics = {
        "mrr":          reciprocal_rank(results, relevant_set),
        f"hit@{k}":     hit_rate_at_k(results, relevant_set, k),
        f"p@{k}":       precision_at_k(results, relevant_set, k),
        f"r@{k}":       recall_at_k(results, relevant_set, k),
        f"ndcg@{k}":    ndcg_at_k(results, relevant_set, k),
    }

    retrieved_clips = [
        {"movie": r.get("movie", ""), "clip_id": r.get("clip_id", ""), "rank": i + 1}
        for i, r in enumerate(results)
    ]

    return {
        "id":              query_obj.get("id", "?"),
        "query":           query_text,
        "metrics":         metrics,
        "intent_correct":  intent_correct,
        "routed_intent":   routed_intent,
        "expected_intent": expected_intent,
        "latency_s":       round(latency, 3),
        "expected_clips":  relevant_clips,
        "retrieved_clips": retrieved_clips,
        "notes":           query_obj.get("notes", ""),
    }


# ===========================================================================
# Aggregate metrics
# ===========================================================================

def aggregate(per_query_results: list[dict], k: int) -> dict:
    """
    Average all per-query metrics and compute intent accuracy.

    Args:
        per_query_results: Output of evaluate_query() for every query.
        k:                 Metric cutoff (used to name the keys).

    Returns:
        Dict of aggregated metric names → float values.
    """
    n = len(per_query_results)
    if n == 0:
        return {}

    metric_keys = [
        "mrr", f"hit@{k}", f"p@{k}", f"r@{k}", f"ndcg@{k}"
    ]

    agg: dict[str, float] = {}
    for key in metric_keys:
        agg[key] = sum(r["metrics"][key] for r in per_query_results) / n

    # Intent accuracy (only for queries that have an expected_intent)
    intent_rows = [r for r in per_query_results if r["intent_correct"] is not None]
    if intent_rows:
        agg["intent_accuracy"] = sum(1 for r in intent_rows if r["intent_correct"]) / len(intent_rows)

    agg["avg_latency_s"] = sum(r["latency_s"] for r in per_query_results) / n

    return agg


# ===========================================================================
# Reporting
# ===========================================================================

_METRIC_LABELS = {
    "mrr":              "MRR (Mean Reciprocal Rank)",
    "hit@{k}":          "Hit Rate@{k}   (≥1 correct in top-{k})",
    "p@{k}":            "Precision@{k}",
    "r@{k}":            "Recall@{k}",
    "ndcg@{k}":         "NDCG@{k}",
    "intent_accuracy":  "Intent Routing Accuracy",
    "avg_latency_s":    "Avg Latency (s)",
}


def _label(key: str, k: int) -> str:
    template = _METRIC_LABELS.get(key, key)
    return template.replace("{k}", str(k))


def print_report(agg: dict, per_query: list[dict], k: int) -> None:
    n = len(per_query)
    print("\n" + "═" * 60)
    print(f"  EVALUATION REPORT  ({n} queries, K={k})")
    print("═" * 60)

    # ── Aggregate metrics ──────────────────────────────────────────────────
    core_keys = ["mrr", f"hit@{k}", f"p@{k}", f"r@{k}", f"ndcg@{k}"]
    for key in core_keys:
        if key in agg:
            print(f"  {_label(key, k):<42}  {agg[key]:.4f}")

    if "intent_accuracy" in agg:
        n_intent = sum(1 for r in per_query if r["intent_correct"] is not None)
        print(f"\n  {'Intent Routing Accuracy':<42}  {agg['intent_accuracy']:.4f}  ({n_intent} queries with labels)")

    print(f"\n  {'Avg Latency (s)':<42}  {agg['avg_latency_s']:.3f}")
    print("─" * 60)

    # ── Per-query breakdown ────────────────────────────────────────────────
    print(f"\n  {'ID':<5}  {'MRR':>6}  {'Hit':>5}  {'P@K':>6}  {'NDCG':>6}  {'Intent':>7}  Query")
    print("  " + "─" * 78)
    for r in per_query:
        m = r["metrics"]
        intent_flag = (
            "✓" if r["intent_correct"] is True
            else "✗" if r["intent_correct"] is False
            else "—"
        )
        query_snippet = r["query"][:42] + ("…" if len(r["query"]) > 42 else "")
        print(
            f"  {r['id']:<5}  "
            f"{m['mrr']:>6.3f}  "
            f"{m[f'hit@{k}']:>5.1f}  "
            f"{m[f'p@{k}']:>6.3f}  "
            f"{m[f'ndcg@{k}']:>6.3f}  "
            f"{intent_flag:>7}  "
            f"{query_snippet}"
        )

    # ── Failures ───────────────────────────────────────────────────────────
    failures = [r for r in per_query if r["metrics"]["mrr"] == 0.0]
    if failures:
        print(f"\n  ── Zero-MRR queries ({len(failures)}) ──")
        for r in failures:
            expected = ", ".join(f"{c['movie']}:{c['clip_id']}" for c in r["expected_clips"])
            retrieved = ", ".join(f"{c['movie']}:{c['clip_id']}" for c in r["retrieved_clips"])
            print(f"  [{r['id']}] {r['query'][:55]}")
            print(f"         expected  : {expected}")
            print(f"         retrieved : {retrieved or '(none)'}")
    print("═" * 60)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the movie scene retrieval system."
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=str(DEFAULT_QUERIES_PATH),
        help="Path to the eval_queries.json ground-truth file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=TOP_K,
        help=f"Metric cutoff K (default: {TOP_K} from config.py).",
    )
    # parser.add_argument(
    #     "--use-llm",
    #     action="store_true",
    #     dest="use_llm",
    #     help="Enable the full LLM router + reranker (requires a local GGUF model).",
    # )
    parser.add_argument(
        "--model",
        type=str,
        default=LLM_MODEL_PATH,
        help="Path to the .gguf model file (only used with --use-llm).",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=LLM_N_GPU_LAYERS,
        dest="gpu_layers",
        help="GPU layers to offload (only used with --use-llm).",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=EVAL_RESULTS_PATH,
        dest="save_report",
        help="Optional path to write the full per-query report as JSON.",
    )
    return parser.parse_args()


# ===========================================================================
# Main entry point
# ===========================================================================

def run_evaluation(
    queries_path: str | Path = DEFAULT_QUERIES_PATH,
    total_queries: int | None = None,
    k: int = TOP_K,
    use_llm: bool = False,
    llm_model_path: str = LLM_MODEL_PATH,
    gpu_layers: int = LLM_N_GPU_LAYERS,
    save_report: str | None = None,
) -> dict[str, Any]:
    """
    Programmatic entry point — can be called from notebooks or other scripts.

    Args:
        queries_path:    Path to eval_queries.json.
        k:               Metric cutoff.
        use_llm:         Whether to initialise the LLM router/reranker.
        llm_model_path:  Path to the GGUF model (only if use_llm=True).
        gpu_layers:      GPU offload layers.
        save_report:     If set, write the full report JSON to this path.

    Returns:
        Dict with 'aggregate' and 'per_query' keys.
    """
    # ── Load ground-truth queries ──────────────────────────────────────────
    queries_path = Path(queries_path)
    if not queries_path.exists():
        raise FileNotFoundError(
            f"Query file not found: {queries_path}\n"
            "Create eval_queries.json next to this script."
        )
    with open(queries_path, encoding="utf-8") as f:
        query_objects: list[dict] = json.load(f)
    
    if total_queries is not None:
        total_queries = min(total_queries, len(query_objects))
        query_objects = query_objects[0:total_queries]

    print(f"[Evaluator] Loaded {len(query_objects)} evaluation queries from {queries_path}")

    # ── Initialise pipeline ────────────────────────────────────────────────
    pipeline = RetrievalPipeline(
        llm_model_path=llm_model_path,
        n_gpu_layers=gpu_layers,
        use_llm=use_llm,
    )

    # ── Run evaluation ─────────────────────────────────────────────────────
    per_query_results: list[dict] = []
    for i, qobj in enumerate(query_objects, start=1):
        print(f"[Evaluator] Query {i}/{len(query_objects)}: {qobj['query'][:60]}")
        result = evaluate_query(pipeline, qobj, k=k)
        per_query_results.append(result)
        print(
            f"           MRR={result['metrics']['mrr']:.3f}  "
            f"NDCG@{k}={result['metrics'][f'ndcg@{k}']:.3f}  "
            f"latency={result['latency_s']:.2f}s"
        )

    # ── Aggregate ──────────────────────────────────────────────────────────
    agg = aggregate(per_query_results, k=k)

    # ── Print human-readable report ────────────────────────────────────────
    print_report(agg, per_query_results, k=k)

    # ── Optionally save JSON report ────────────────────────────────────────
    if save_report:
        report_path = Path(save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        full_report = {"k": k, "aggregate": agg, "per_query": per_query_results}
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        print(f"\n[Evaluator] Full report saved to {report_path}")

    return {"aggregate": agg, "per_query": per_query_results}


def main() -> None:
    args = parse_args()
    
    run_evaluation(
        queries_path=args.queries,
        k=args.k,
        use_llm=True,
        llm_model_path=args.model,
        gpu_layers=args.gpu_layers,
        save_report=args.save_report,
    )


if __name__ == "__main__":
    main()
