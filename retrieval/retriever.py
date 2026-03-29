"""
retriever.py — Query the indices and return ranked movies.

Retrieval pipeline:
  1. Entity extraction — scan query for actor/director names in structured indices;
     matched movies receive a strong score boost so they surface even when the
     screenplay text never mentions the actor's real name.
  2. Text search      — FAISS (dense) and/or BM25 (sparse) over scene/plot indices.
  3. RRF fusion       — combine rankings from multiple indices.
  4. Movie aggregation — group scene hits by movie, sum RRF scores.
  5. Entity boost     — add ENTITY_BOOST to aggregated score for entity-matched movies.

Usage:
    python retriever.py --query "the boy receives letters from a magic school"
    python retriever.py --query "what are you listening to" --intent dialogue
    python retriever.py --query "car chase with loud music" --intent scene,full
    python retriever.py --query "..." --top-k 5 --mode bm25
    python retriever.py --query "..." --top-k 5 --mode faiss
    python retriever.py --query "..." --top-k 5 --mode hybrid   (default)
"""

import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR   = Path("../indices")
EMBED_MODEL = "all-MiniLM-L6-v2"
VALID_INTENTS = ("scene", "dialogue", "character", "full", "plot")

# Score added to any movie whose actor/director appears in the query.
# RRF scores typically range 0.01–0.10, so 0.5 reliably surfaces entity matches
# while still allowing text scores to break ties among multiple entity matches.
ENTITY_BOOST = 0.5

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def tokenize(text: str) -> list:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def search_faiss(query: str, index_name: str, top_k: int = 20) -> list:
    """Returns list of (score, meta_dict) sorted by descending score."""
    index_path = INDEX_DIR / f"{index_name}.faiss"
    meta_path  = INDEX_DIR / f"{index_name}_meta.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    index = faiss.read_index(str(index_path))
    meta  = json.loads(meta_path.read_text(encoding="utf-8"))

    q_vec = _get_model().encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_vec, min(top_k, index.ntotal))

    return [(float(scores[0][i]), meta[indices[0][i]]) for i in range(len(indices[0]))]


def search_bm25(query: str, index_name: str, top_k: int = 20) -> list:
    """Returns list of (score, meta_dict) sorted by descending score."""
    bm25_path = INDEX_DIR / f"{index_name}_bm25.pkl"

    if not bm25_path.exists():
        raise FileNotFoundError(f"BM25 index not found: {bm25_path}")

    with open(bm25_path, "rb") as f:
        data = pickle.load(f)

    bm25   = data["bm25"]
    meta   = data["meta"]
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(float(scores[i]), meta[i]) for i in ranked]


def rrf_fuse(rankings: list[list], k: int = 60) -> list:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.
    Returns meta dicts sorted by fused RRF score.
    """
    rrf_scores: dict = {}
    meta_lookup: dict = {}

    for ranking in rankings:
        for rank, (_, meta) in enumerate(ranking):
            sid = meta["movie_id"] if "overview" in meta else meta["scene_id"]
            rrf_scores[sid] = rrf_scores.get(sid, 0.0) + 1.0 / (k + rank + 1)
            meta_lookup[sid] = meta

    fused = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return [{"rrf_score": rrf_scores[sid], **meta_lookup[sid]} for sid in fused]


def aggregate_by_movie(fused_results: list, top_scenes: int = 20) -> list:
    """
    Group top scene hits by movie_id.
    Movie score = sum of RRF scores of its scenes (rewards multiple hits).
    """
    movie_scores: dict = defaultdict(float)
    movie_scenes: dict = defaultdict(list)

    for hit in fused_results[:top_scenes]:
        mid = hit["movie_id"]
        movie_scores[mid] += hit["rrf_score"]
        movie_scenes[mid].append(hit)

    ranked = sorted(movie_scores, key=movie_scores.get, reverse=True)
    return [
        {
            "movie_id":   mid,
            "score":      round(movie_scores[mid], 4),
            "top_scenes": movie_scenes[mid][:3],
        }
        for mid in ranked
    ]


def extract_entities(query: str) -> tuple[set, list]:
    """
    Scan the query for actor and director names from the structured indices.

    Matches require all name tokens to appear in the query (so "Woody Harrelson"
    matches only if both "woody" and "harrelson" are present). Single-token names
    are skipped to avoid false positives on common words.

    Returns (matched_movie_ids, matched_names).
    """
    query_tokens = set(tokenize(query))
    matched_ids: set = set()
    matched_names: list = []

    for index_file in ("actor_index.json", "director_index.json"):
        path = INDEX_DIR / index_file
        if not path.exists():
            continue
        index = json.loads(path.read_text(encoding="utf-8"))
        for name, movie_ids in index.items():
            name_tokens = set(tokenize(name))
            if len(name_tokens) < 2:
                continue  # skip single-token names to avoid false positives
            if name_tokens.issubset(query_tokens):
                matched_ids.update(movie_ids)
                matched_names.append(name)

    return matched_ids, matched_names


def _load_registry() -> dict:
    path = INDEX_DIR / "registry.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_genre_index() -> dict:
    path = INDEX_DIR / "genre_index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _build_allowed_ids(
    genre_filter: str | None,
    year_range: tuple | None,
) -> set | None:
    """
    Return the set of movie_ids that satisfy genre/year constraints,
    or None if no constraints are active.
    Used for pre-filtering before text search so constrained movies
    are included in fusion even when they rank low on text similarity.
    """
    if not genre_filter and not year_range:
        return None

    registry   = _load_registry()
    genre_ids: set | None = None

    if genre_filter:
        genre_index = _load_genre_index()
        target = genre_filter.lower()
        matched: set = set()
        for g, ids in genre_index.items():
            if target in g or g in target:
                matched.update(ids)
        genre_ids = matched if matched else None  # None = no genre match → skip genre constraint

    allowed: set | None = None
    if genre_ids is not None:
        allowed = genre_ids

    if year_range:
        yr_min, yr_max = year_range
        year_ids: set = set()
        for mid, info in registry.items():
            try:
                year = int(info.get("year", 0))
                if yr_min and year < yr_min:
                    continue
                if yr_max and year > yr_max:
                    continue
                year_ids.add(mid)
            except (ValueError, TypeError):
                year_ids.add(mid)  # unknown year → include
        allowed = (allowed & year_ids) if allowed is not None else year_ids

    return allowed  # None means no constraint


def _apply_exclusions(results: list, exclude_ids: set | None) -> list:
    """Remove explicitly excluded movies from results (negation queries)."""
    if not exclude_ids:
        return results
    filtered = [r for r in results if r["movie_id"] not in exclude_ids]
    return filtered if filtered else results  # fail-open if everything excluded


def retrieve(
    query: str,
    intents: list | None = None,
    mode: str = "hybrid",
    top_k: int = 10,
    genre_filter: str | None = None,
    year_range: tuple | None = None,
    exclude_ids: set | None = None,
) -> tuple[list, list]:
    """
    Args:
        query:        natural-language query (may be a rewritten version)
        intents:      list of index names to search (default: ["full"])
        mode:         "faiss" | "bm25" | "hybrid" (default)
        top_k:        number of candidates per index before fusion
        genre_filter: genre name to restrict results (fuzzy match)
        year_range:   (year_min, year_max) tuple; either element may be None
        exclude_ids:  set of movie_ids to remove from results

    Returns:
        (results, entity_names)
        results      — list of {movie_id, score, top_scenes} ranked by relevance
        entity_names — actor/director names detected in the query (for display)
    """
    if intents is None:
        intents = ["full"]

    # Pre-build allowed movie_id set from genre/year constraints so that
    # text search results are restricted to eligible movies before fusion.
    # This ensures constrained movies appear in results even if they rank
    # low on text similarity for the whole corpus.
    allowed_ids = _build_allowed_ids(genre_filter, year_range)
    has_filters = allowed_ids is not None

    # Larger search pool when filtering: we may need many raw hits to find
    # enough matches within the constrained set.
    pool_mult = 8 if has_filters else 2

    # Text search + pre-filter
    all_rankings = []
    for intent in intents:
        if mode in ("faiss", "hybrid"):
            hits = search_faiss(query, intent, top_k * pool_mult)
            if allowed_ids is not None:
                hits = [(s, m) for s, m in hits if m.get("movie_id") in allowed_ids]
            all_rankings.append(hits)
        if mode in ("bm25", "hybrid"):
            hits = search_bm25(query, intent, top_k * pool_mult)
            if allowed_ids is not None:
                hits = [(s, m) for s, m in hits if m.get("movie_id") in allowed_ids]
            all_rankings.append(hits)

    fused   = rrf_fuse(all_rankings)
    results = aggregate_by_movie(fused, top_scenes=top_k * len(intents))

    # Entity boost: actor/director names found in query -> boost matched movies.
    # Also inject any entity-matched movies cut off during aggregation.
    entity_ids, entity_names = extract_entities(query)
    if entity_ids:
        present_ids = {r["movie_id"] for r in results}
        for r in results:
            if r["movie_id"] in entity_ids:
                r["score"] = round(r["score"] + ENTITY_BOOST, 4)
        for mid in entity_ids:
            if mid not in present_ids:
                results.append({"movie_id": mid, "score": ENTITY_BOOST, "top_scenes": []})
        results.sort(key=lambda r: r["score"], reverse=True)

    results = _apply_exclusions(results, exclude_ids)

    return results, entity_names


def _load_title_map(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {entry["tmdb_id"]: entry["title"] for entry in manifest.values()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Query the screenplay retrieval indices.")
    ap.add_argument("--query",      required=True)
    ap.add_argument("--intent",     default="full",
                    help="Comma-separated intents: scene,dialogue,character,full,plot (default: full)")
    ap.add_argument("--mode",       default="hybrid", choices=["faiss", "bm25", "hybrid"])
    ap.add_argument("--top-k",      type=int, default=5)
    ap.add_argument("--index-dir",  default="../indices")
    ap.add_argument("--manifest",   default="../screenplays/manifest.json")
    args = ap.parse_args()

    global INDEX_DIR
    INDEX_DIR = Path(args.index_dir)

    intents = [i.strip() for i in args.intent.split(",") if i.strip() in VALID_INTENTS]
    if not intents:
        print(f"Invalid intents. Choose from: {VALID_INTENTS}", file=sys.stderr)
        sys.exit(1)

    title_map = _load_title_map(Path(args.manifest))
    results, entity_names = retrieve(args.query, intents=intents, mode=args.mode, top_k=args.top_k)

    width = 72
    print()
    print("=" * width)
    print(f"  Query  : {args.query[:width - 12]}")
    print(f"  Intent : {', '.join(intents)}   Mode : {args.mode}")
    if entity_names:
        print(f"  Entities: {', '.join(entity_names)}")
    print("=" * width)

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
            print(f"    text   : {snippet}{'...' if len(s.get('text', s.get('overview', ''))) > 100 else ''}")
            print()


if __name__ == "__main__":
    main()
