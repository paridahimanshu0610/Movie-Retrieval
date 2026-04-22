"""
tmdb_fetch.py — Fetch TMDB metadata and Wikipedia plot summaries for all movies in manifest.json.

For each entry:
  1. If tmdb_id is "0", searches TMDB by title (+ year from filename) and resolves the ID.
  2. Fetches full movie details: overview, genres, cast, director, release date, poster URL.
  3. Fetches the Plot section from the corresponding Wikipedia article.
  4. Writes one metadata JSON per movie to --output-dir.
  5. Updates manifest.json in-place with resolved tmdb_ids.

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

TMDB_BASE       = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
WIKI_API        = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS    = {"User-Agent": "ScreenplayRetrieval/1.0 (academic project)"}


# ── TMDB helpers ──────────────────────────────────────────────────────────────

def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _get(url: str, params: dict, token: str, retries: int = 3) -> dict:
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
    m = _YEAR_RE.search(filename)
    return m.group(1) if m else None


def search_movie(title: str, year: str | None, token: str) -> dict | None:
    params = {"query": title, "include_adult": False}
    if year:
        params["year"] = year

    data = _get(f"{TMDB_BASE}/search/movie", params, token)
    results = data.get("results", [])

    if not results and year:
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
    details = _get(
        f"{TMDB_BASE}/movie/{tmdb_id}",
        {"append_to_response": "credits"},
        token,
    )

    crew      = details.get("credits", {}).get("crew", [])
    directors = [p["name"] for p in crew if p.get("job") == "Director"]
    cast      = details.get("credits", {}).get("cast", [])
    top_cast  = [{"name": p["name"], "character": p.get("character", "")} for p in cast]
    poster    = details.get("poster_path")

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
        "poster_url":        f"{TMDB_IMAGE_BASE}{poster}" if poster else None,
        "tmdb_url":          f"https://www.themoviedb.org/movie/{details['id']}",
    }


# ── Wikipedia helpers ─────────────────────────────────────────────────────────

_WIKI_MAX_WAIT = 30  # never wait more than 30s; skip and let the caller retry later

def _wiki_get(params: dict, retries: int = 3) -> dict:
    params.setdefault("format", "json")
    for attempt in range(retries):
        resp = requests.get(WIKI_API, params=params, headers=WIKI_HEADERS, timeout=10)
        if resp.status_code == 429:
            requested = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
            wait = min(requested, _WIKI_MAX_WAIT)
            logger.warning("Wikipedia rate limited (Retry-After=%ds), waiting %ds (attempt %d)...",
                           requested, wait, attempt + 1)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Wikipedia API failed after {retries} attempts")


def _strip_wikitext(text: str) -> str:
    """Convert wikitext to plain text."""
    # Remove <ref> tags and their content
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/?>", "", text)
    # Remove nested templates {{...}} iteratively
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    # [[File:...]] and [[Image:...]] blocks
    text = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    # [[link|display]] → display, [[link]] → link
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    # Bold/italic markup
    text = re.sub(r"'{2,3}", "", text)
    # Section headers
    text = re.sub(r"==+[^=]*==+", "", text)
    # HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_wiki_page(title: str, year: str | None) -> str | None:
    """
    Search Wikipedia for the right film page.
    Returns the page title string or None if not found.
    """
    candidates = []
    if year:
        candidates.append(f"{title} ({year} film)")
    candidates.append(f"{title} (film)")
    candidates.append(title)

    title_lower = title.lower()

    # Try direct page lookup for common disambiguation patterns first
    for candidate in candidates:
        data = _wiki_get({
            "action":    "query",
            "titles":    candidate,
            "redirects": 1,
        })
        pages = data.get("query", {}).get("pages", {})
        if pages and "-1" not in pages:
            page = next(iter(pages.values()))
            return page.get("title")

    # Fall back to search
    query = f"{title} film"
    if year:
        query = f"{title} {year} film"

    data = _wiki_get({
        "action":      "query",
        "list":        "search",
        "srsearch":    query,
        "srnamespace": 0,
        "srlimit":     5,
    })
    results = data.get("query", {}).get("search", [])
    for r in results:
        page_title = r.get("title", "")
        if title_lower in page_title.lower():
            return page_title

    return results[0]["title"] if results else None


def _fetch_wiki_plot(page_title: str) -> str | None:
    """
    Fetch the Plot section from a Wikipedia film article.
    Returns plain text or None if no plot section found.
    """
    # Get section list
    data = _wiki_get({"action": "parse", "page": page_title, "prop": "sections"})
    sections = data.get("parse", {}).get("sections", [])

    plot_idx = None
    for s in sections:
        if s.get("line", "").lower() in ("plot", "plot summary", "synopsis"):
            plot_idx = s["index"]
            break

    if plot_idx is None:
        return None

    # Fetch wikitext for that section
    data = _wiki_get({
        "action":  "parse",
        "page":    page_title,
        "prop":    "wikitext",
        "section": plot_idx,
    })
    wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
    if not wikitext:
        return None

    text = _strip_wikitext(wikitext)
    return text if len(text) > 100 else None


def fetch_wiki_plot(title: str, year: str | None) -> str | None:
    """Public entry point: search for the film page and return its plot text."""
    try:
        page_title = _find_wiki_page(title, year)
        if not page_title:
            logger.warning("  [Wiki] Page not found for '%s'", title)
            return None
        plot = _fetch_wiki_plot(page_title)
        if plot:
            logger.info("  [Wiki] Fetched plot (%d chars) from '%s'", len(plot), page_title)
        else:
            logger.warning("  [Wiki] No plot section in '%s'", page_title)
        return plot
    except Exception as e:
        logger.warning("  [Wiki] Failed for '%s': %s", title, e)
        return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

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
        year    = _year_from_filename(filename)

        if tmdb_id == "0":
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

        # TMDB metadata
        try:
            meta = fetch_movie_metadata(tmdb_id, token)
        except Exception as e:
            logger.error("  Metadata fetch failed for %s (id=%s): %s", title, tmdb_id, e)
            results[slug] = {"tmdb_id": tmdb_id, "title": title, "error": str(e)}
            continue

        # Wikipedia plot — skip if already present in existing metadata file
        out_path = os.path.join(output_dir, f"{slug}_metadata.json")
        existing_wiki_plot = None
        if os.path.isfile(out_path):
            try:
                with open(out_path, encoding="utf-8") as f:
                    existing = json.load(f)
                existing_wiki_plot = existing.get("wiki_plot")
            except Exception:
                pass

        if existing_wiki_plot:
            logger.info("  [Wiki] Already have plot for '%s', skipping", title)
            meta["wiki_plot"] = existing_wiki_plot
        elif not dry_run:
            # Use canonical TMDB title + year for Wikipedia — more reliable than manifest title
            tmdb_title = meta.get("title") or title
            tmdb_year = meta.get("release_date", "")[:4] or year
            meta["wiki_plot"] = fetch_wiki_plot(tmdb_title, tmdb_year)
            if meta["wiki_plot"]:
                time.sleep(1.0)  # be polite to Wikipedia between successful fetches

        results[slug] = meta

        if not dry_run:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            logger.info("  Written: %s_metadata.json", slug)

        time.sleep(0.26)  # TMDB rate limit: 40 req/10s

    if manifest_updated and not dry_run:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info("manifest.json updated with resolved tmdb_ids")

    found     = sum(1 for m in results.values() if m.get("tmdb_id", "0") != "0" and "error" not in m)
    not_found = sum(1 for m in results.values() if "error" in m)
    wiki_ok   = sum(1 for m in results.values() if m.get("wiki_plot"))
    print(f"\nDone. {len(results)} movies — {found} fetched, {not_found} failed, {wiki_ok} with wiki plot.")

    if dry_run:
        print("\nDry-run results:")
        for slug, meta in results.items():
            status = meta.get("error", "ok")
            print(f"  {slug:<50} tmdb_id={meta.get('tmdb_id','?'):>8}  [{status}]")


def main():
    ap = argparse.ArgumentParser(description="Fetch TMDB metadata and Wikipedia plots for all manifest movies.")
    ap.add_argument("--manifest",   default="../screenplays/manifest.json")
    ap.add_argument("--output-dir", default="../output")
    ap.add_argument("--env",        default="../.env")
    ap.add_argument("--dry-run",    action="store_true",
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
