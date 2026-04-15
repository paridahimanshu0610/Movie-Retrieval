"""
tmdb_fetch.py — Fetch TMDB metadata for all movies in manifest.json.

For each entry:
  1. If tmdb_id is "0", searches TMDB by title (+ year from filename if present)
     and resolves to the correct ID.
  2. Fetches full movie details: overview, genres, cast, director, release date,
     runtime, original language, poster URL.
  3. Writes one metadata JSON per movie to --output-dir.
  4. Updates manifest.json in-place with the resolved tmdb_ids.

Usage:
    python tmdb_fetch.py \\
        --manifest  ../screenplays/manifest.json \\
        --output-dir ../output \\
        --env       ../.env

    # Dry-run (search + print, no file writes):
    python tmdb_fetch.py --manifest ../screenplays/manifest.json --dry-run
"""

import argparse
import json
import logging
import os
import re
import sys
import time

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                    stream=sys.stderr)
logger = logging.getLogger(__name__)

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


def _get(url: str, params: dict, token: str, retries: int = 3) -> dict:
    """GET with retry on rate-limit (429) or transient errors."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=_headers(token), timeout=10)
        except requests.RequestException as e:
            logger.warning("Request error (attempt %d): %s", attempt + 1, e)
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 5))
            logger.warning("Rate limited, waiting %ds...", wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Failed after {retries} attempts: {url}")


_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")

def _year_from_filename(filename: str) -> str | None:
    """Extract 4-digit year from a filename like 'baby-driver-2017.pdf'."""
    m = _YEAR_RE.search(filename)
    return m.group(1) if m else None


def search_movie(title: str, year: str | None, token: str) -> dict | None:
    """
    Search TMDB for a movie by title (and optionally year).
    Returns the best-matching result dict, or None if not found.
    Prefers exact title match; falls back to the top popularity result.
    """
    params = {"query": title, "include_adult": False}
    if year:
        params["year"] = year

    data = _get(f"{TMDB_BASE}/search/movie", params, token)
    results = data.get("results", [])

    if not results and year:
        # Retry without year in case of off-by-one in the filename
        params.pop("year")
        data = _get(f"{TMDB_BASE}/search/movie", params, token)
        results = data.get("results", [])

    if not results:
        return None

    title_lower = title.lower()
    for r in results:
        if r.get("title", "").lower() == title_lower:
            return r
    return results[0]


def fetch_movie_metadata(tmdb_id: str, token: str) -> dict:
    """Fetch full movie metadata from TMDB including credits."""
    details = _get(
        f"{TMDB_BASE}/movie/{tmdb_id}",
        {"append_to_response": "credits"},
        token,
    )

    crew = details.get("credits", {}).get("crew", [])
    directors = [p["name"] for p in crew if p.get("job") == "Director"]

    cast = details.get("credits", {}).get("cast", [])
    top_cast = [
        {"name": p["name"], "character": p.get("character", "")}
        for p in cast
    ]

    poster_path = details.get("poster_path")

    return {
        "tmdb_id":           str(details["id"]),
        "title":             details.get("title", ""),
        "original_title":    details.get("original_title", ""),
        "overview":          details.get("overview", ""),
        "release_date":      details.get("release_date", ""),
        "runtime_minutes":   details.get("runtime"),
        "genres":            [g["name"] for g in details.get("genres", [])],
        "original_language": details.get("original_language", ""),
        "popularity":        details.get("popularity"),
        "vote_average":      details.get("vote_average"),
        "vote_count":        details.get("vote_count"),
        "directors":         directors,
        "cast":              top_cast,
        "poster_url":        f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None,
        "tmdb_url":          f"https://www.themoviedb.org/movie/{details['id']}",
    }


def run(manifest_path: str, output_dir: str, token: str, dry_run: bool = False):
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)

    manifest_updated = False
    results = {}

    for filename, entry in manifest.items():
        slug    = entry["slug"]
        title   = entry["title"]
        tmdb_id = entry.get("tmdb_id", "0")

        if tmdb_id == "0":
            year = _year_from_filename(filename)
            logger.info("Searching TMDB: '%s' (year=%s)", title, year or "?")
            result = search_movie(title, year, token)

            if not result:
                logger.warning("  NOT FOUND on TMDB: %s", title)
                results[slug] = {"tmdb_id": "0", "title": title, "error": "not_found"}
                continue

            tmdb_id = str(result["id"])
            logger.info("  Found: %s (%s) -> tmdb_id=%s",
                        result.get("title"), result.get("release_date", "")[:4], tmdb_id)

            if not dry_run:
                entry["tmdb_id"] = tmdb_id
                manifest_updated = True
        else:
            logger.info("Using existing tmdb_id=%s for '%s'", tmdb_id, title)

        try:
            meta = fetch_movie_metadata(tmdb_id, token)
        except Exception as e:
            logger.error("  Metadata fetch failed for %s (id=%s): %s", title, tmdb_id, e)
            results[slug] = {"tmdb_id": tmdb_id, "title": title, "error": str(e)}
            continue

        results[slug] = meta

        if not dry_run:
            out_path = os.path.join(output_dir, f"{slug}_metadata.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            logger.info("  Written: %s_metadata.json", slug)

        # Stay within TMDB rate limits (40 req/10s)
        time.sleep(0.26)

    if manifest_updated and not dry_run:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info("manifest.json updated with resolved tmdb_ids")

    found     = sum(1 for m in results.values() if m.get("tmdb_id", "0") != "0" and "error" not in m)
    not_found = sum(1 for m in results.values() if "error" in m)
    print(f"\nDone. {len(results)} movies -- {found} fetched, {not_found} failed.")

    if dry_run:
        print("\nDry-run results:")
        for slug, meta in results.items():
            status = meta.get("error", "ok")
            print(f"  {slug:<50} tmdb_id={meta.get('tmdb_id','?'):>8}  [{status}]")


def main():
    ap = argparse.ArgumentParser(description="Fetch TMDB metadata for all manifest movies.")
    ap.add_argument("--manifest",    default="../screenplays/manifest.json")
    ap.add_argument("--output-dir",  default="../output")
    ap.add_argument("--env",         default="../.env")
    ap.add_argument("--dry-run",     action="store_true",
                    help="Search and print results without writing any files")
    args = ap.parse_args()

    env_path = os.path.abspath(args.env)
    if os.path.isfile(env_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip())

    token = os.environ.get("TMDB_API_ACCESS_TOKEN")
    if not token:
        sys.exit("TMDB_API_ACCESS_TOKEN not found. Set it in .env or as an environment variable.")

    run(
        manifest_path=os.path.abspath(args.manifest),
        output_dir=os.path.abspath(args.output_dir),
        token=token,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
