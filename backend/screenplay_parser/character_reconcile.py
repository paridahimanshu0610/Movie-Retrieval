"""
character_reconcile.py — Match screenplay character names to TMDB cast entries.

For each movie, collects all unique ALL-CAPS character names from scene JSON,
then tries to match them to TMDB cast (character name + actor name).

Matching strategy (in priority order):
  1. Exact       — "JACK SPARROW" == "jack sparrow"
  2. Token-exact — every token in screenplay name appears in TMDB name
                   ("SPARROW" matches "Jack Sparrow")
  3. Substring   — screenplay name is a substring of TMDB name or vice-versa
  4. Unmatched   — flagged for manual review

Output per movie:  output/{slug}_characters.json
Schema:
    [
        {
            "screenplay_name":  "JACK SPARROW",
            "tmdb_character":   "Jack Sparrow",
            "actor":            "Johnny Depp",
            "match_type":       "exact",        // exact|token|substring|unmatched
            "scene_count":      12,             // scenes this character appears in
            "tmdb_order":       0               // billing order in TMDB cast list
        },
        ...
    ]

Usage:
    python character_reconcile.py \\
        --manifest   ../screenplays/manifest.json \\
        --scenes-dir ../output \\
        --output-dir ../output

    # Single movie
    python character_reconcile.py --slug pulp_fiction \\
        --manifest ../screenplays/manifest.json \\
        --scenes-dir ../output --output-dir ../output
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                    stream=sys.stderr)
logger = logging.getLogger(__name__)

_SUFFIX_RE = re.compile(
    r"\s*\((?:V\.O\.?|O\.S\.?|O\.C\.?|CONT['']D|VOICE|ARCHIVE|PRE-LAP)\)\s*$",
    re.IGNORECASE,
)


def _normalise(name: str) -> str:
    name = _SUFFIX_RE.sub("", name).strip()
    return re.sub(r"\s+", " ", name.lower())


def _tokens(name: str) -> set:
    return set(re.findall(r"[a-z]+", _normalise(name)))


def _match(sp_name: str, cast: list) -> dict | None:
    """
    Try to match one screenplay name against the TMDB cast list.
    Returns the best match dict or None.
    """
    sp_norm   = _normalise(sp_name)
    sp_tokens = _tokens(sp_name)

    best = None  # (priority, cast_entry, match_type)

    for entry in cast:
        tmdb_char   = entry.get("character", "")
        tmdb_norm   = _normalise(tmdb_char)
        tmdb_tokens = _tokens(tmdb_char)

        if sp_norm == tmdb_norm:
            return {**entry, "match_type": "exact"}

        if sp_tokens and sp_tokens.issubset(tmdb_tokens):
            candidate = (1, entry, "token")
        elif sp_norm and (sp_norm in tmdb_norm or tmdb_norm in sp_norm):
            candidate = (2, entry, "substring")
        else:
            continue

        if best is None or candidate[0] < best[0]:
            best = candidate

    if best:
        _, entry, match_type = best
        return {**entry, "match_type": match_type}

    return None


def reconcile_movie(slug: str, scenes_dir: str, output_dir: str) -> list:
    scenes_path   = os.path.join(scenes_dir,  f"{slug}_scenes.json")
    metadata_path = os.path.join(scenes_dir,  f"{slug}_metadata.json")
    output_path   = os.path.join(output_dir,  f"{slug}_characters.json")

    if not os.path.exists(scenes_path):
        logger.warning("Scenes file not found: %s", scenes_path)
        return []
    if not os.path.exists(metadata_path):
        logger.warning("Metadata file not found: %s", metadata_path)
        return []

    scenes   = json.load(open(scenes_path,   encoding="utf-8"))
    metadata = json.load(open(metadata_path, encoding="utf-8"))
    cast     = metadata.get("cast", [])

    cast_with_order = [
        {**entry, "tmdb_order": i}
        for i, entry in enumerate(cast)
    ]

    scene_counts: Counter = Counter()
    for scene in scenes:
        for char in scene.get("characters", []):
            scene_counts[char] += 1

    results = []
    unmatched = []

    for sp_name, count in sorted(scene_counts.items(), key=lambda x: -x[1]):
        match = _match(sp_name, cast_with_order)

        if match:
            results.append({
                "screenplay_name": sp_name,
                "tmdb_character":  match.get("character", ""),
                "actor":           match.get("name", ""),
                "match_type":      match["match_type"],
                "scene_count":     count,
                "tmdb_order":      match.get("tmdb_order", -1),
            })
        else:
            unmatched.append({
                "screenplay_name": sp_name,
                "tmdb_character":  None,
                "actor":           None,
                "match_type":      "unmatched",
                "scene_count":     count,
                "tmdb_order":      -1,
            })

    all_results = results + unmatched

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    matched_count   = len(results)
    unmatched_count = len(unmatched)
    match_rate      = matched_count / len(all_results) * 100 if all_results else 0

    logger.info(
        "[OK] %-52s  %3d matched  %3d unmatched  (%.0f%%)",
        slug, matched_count, unmatched_count, match_rate,
    )

    return all_results


def main():
    ap = argparse.ArgumentParser(
        description="Reconcile screenplay character names with TMDB cast."
    )
    ap.add_argument("--manifest",    default="../screenplays/manifest.json")
    ap.add_argument("--scenes-dir",  default="../output")
    ap.add_argument("--output-dir",  default="../output")
    ap.add_argument("--slug",        default=None,
                    help="Process a single movie slug instead of all")
    ap.add_argument("--incremental", action="store_true",
                    help="Skip movies whose characters file already exists.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    slugs = [e["slug"] for e in manifest.values()]

    if args.slug:
        if args.slug not in slugs:
            sys.exit(f"Slug '{args.slug}' not found in manifest.")
        slugs = [args.slug]

    total_matched = total_unmatched = 0
    for slug in slugs:
        if args.incremental:
            output_path = os.path.join(args.output_dir, f"{slug}_characters.json")
            if os.path.exists(output_path):
                logger.info("[SKIP] %s: already reconciled", slug)
                continue

        results = reconcile_movie(slug, args.scenes_dir, args.output_dir)
        total_matched   += sum(1 for r in results if r["match_type"] != "unmatched")
        total_unmatched += sum(1 for r in results if r["match_type"] == "unmatched")

    total = total_matched + total_unmatched
    rate  = total_matched / total * 100 if total else 0
    print(f"\nDone. {total} characters -- {total_matched} matched, "
          f"{total_unmatched} unmatched ({rate:.1f}% match rate)")


if __name__ == "__main__":
    main()
