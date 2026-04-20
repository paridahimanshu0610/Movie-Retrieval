"""
Reciprocal Rank Fusion (RRF) for merging multiple ranked result lists.

RRF Formula
-----------
    RRF_score(d) = Σ  1 / (k + rank(d, list_i))

where rank is 1-based and k is a smoothing constant (default 60, from
the original Cormack et al. 2009 paper).

Why RRF instead of score normalisation?
----------------------------------------
BM25 and cosine similarity live on different scales.  Normalising each
list independently before combining inflates scores from short lists and
is sensitive to outliers.  RRF only cares about *rank position*, so it
is robust to scale mismatches and consistently outperforms linear
combination in multi-retriever settings.
"""

from __future__ import annotations

from collections import defaultdict

from config import RRF_K


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = RRF_K,
    top_k: int | None = None,
) -> list[dict]:
    """
    Merge multiple ranked result lists into a single fused ranking.

    Each result dict must contain 'movie' and 'clip_id' keys for
    de-duplication.  The first occurrence of a (movie, clip_id) pair
    is used to carry forward all metadata; additional occurrences only
    contribute their rank score.

    Args:
        result_lists: List of ranked result lists (each from one retriever
                      or one field query).
        k:            RRF smoothing constant.
        top_k:        If provided, truncate the output to this length.

    Returns:
        Deduplicated, re-ranked list with an added 'rrf_score' key.
    """
    scores: dict[str, float] = defaultdict(float)
    meta_store: dict[str, dict] = {}
    # Track which field types contributed to each clip for transparency.
    contributing_fields: dict[str, list[str]] = defaultdict(list)

    for result_list in result_lists:
        for rank_0based, result in enumerate(result_list):
            key = _clip_key(result)
            rrf_score = 1.0 / (k + rank_0based + 1)  # rank is 1-based internally
            scores[key] += rrf_score

            if key not in meta_store:
                meta_store[key] = result.copy()

            field = result.get("field_type", "unknown")
            if field not in contributing_fields[key]:
                contributing_fields[key].append(field)

    # Sort by fused score
    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)

    fused = []
    for key in sorted_keys:
        entry = meta_store[key].copy()
        entry["rrf_score"] = round(scores[key], 6)
        entry["matched_fields"] = contributing_fields[key]
        fused.append(entry)

    return fused[:top_k] if top_k is not None else fused


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clip_key(result: dict) -> str:
    """Stable dedup key: 'movie__clip_id'."""
    return f"{result.get('movie', '')}__{ result.get('clip_id', '')}"