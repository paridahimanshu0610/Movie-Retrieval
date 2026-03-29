"""
manifest_sync.py — Sync manifest.json with the screenplays directory.

Scans the input directory for supported screenplay files (.pdf, .html, .txt),
compares against the existing manifest, and adds entries for any new files.
Existing entries are never modified or removed.

Title and slug are derived automatically from the filename:
    "inglourious-bastards-2009.pdf"
        -> title: "Inglourious Basterds"   (title-cased, year stripped)
        -> slug:  "inglourious_basterds"

New entries get tmdb_id "0" -- run tmdb_fetch.py afterward to resolve them.

Usage:
    python manifest_sync.py
        --screenplays-dir ../screenplays
        --manifest        ../screenplays/manifest.json

    # Dry-run: print what would change without writing
    python manifest_sync.py --screenplays-dir ../screenplays --dry-run
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                    stream=sys.stderr)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt"}

_LOWERCASE_WORDS = frozenset([
    "a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet",
    "at", "by", "in", "of", "on", "to", "up", "as", "is",
])

# Known title corrections where auto-derivation produces the wrong result.
_OVERRIDES = {
    "inglourious-bastards": ("Inglourious Basterds", "inglourious_basterds"),
    "harry-potter-and-the-deathly-hollows-part-2": (
        "Harry Potter and the Deathly Hallows Part 2",
        "harry_potter_and_the_deathly_hallows_part_2",
    ),
}

_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def _filename_to_title_slug(filename: str) -> tuple[str, str]:
    """
    Derive (title, slug) from a screenplay filename.

    Examples:
        "baby-driver-2017.pdf"     -> ("Baby Driver", "baby_driver")
        "kill-bill-vol-1-2003.pdf" -> ("Kill Bill Vol 1", "kill_bill_vol_1")
        "2012.html"                -> ("2012", "2012")
    """
    stem = Path(filename).stem.lower()

    if stem in _OVERRIDES:
        return _OVERRIDES[stem]

    stem_no_year = _YEAR_RE.sub("", stem).strip("-")
    if stem_no_year in _OVERRIDES:
        return _OVERRIDES[stem_no_year]

    words_no_year = _YEAR_RE.sub("", stem).strip("-").replace("-", " ").split()
    slug = "_".join(w for w in words_no_year if w)

    def _cap(word: str, i: int, total: int) -> str:
        if i == 0 or i == total - 1:
            return word.capitalize()
        return word if word in _LOWERCASE_WORDS else word.capitalize()

    total = len(words_no_year)
    title = " ".join(_cap(w, i, total) for i, w in enumerate(words_no_year))

    if not title:
        title = stem
        slug  = stem

    return title, slug


def sync(screenplays_dir: str, manifest_path: str, dry_run: bool = False) -> int:
    """
    Scan screenplays_dir, add missing entries to manifest.
    Returns the number of new entries added.
    """
    screenplays_dir = Path(screenplays_dir).resolve()
    manifest_path   = Path(manifest_path).resolve()

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest: dict = json.load(f)
    else:
        manifest = {}
        logger.info("manifest.json not found, will create: %s", manifest_path)

    existing_files = set(manifest.keys())
    existing_slugs = {e["slug"] for e in manifest.values()}

    candidates = sorted(
        p for p in screenplays_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    new_entries: dict = {}

    for path in candidates:
        filename = path.name
        if filename in existing_files:
            continue

        title, slug = _filename_to_title_slug(filename)

        # Append _2, _3 etc. to resolve duplicate slugs
        base_slug = slug
        suffix = 2
        while slug in existing_slugs or slug in {e["slug"] for e in new_entries.values()}:
            slug = f"{base_slug}_{suffix}"
            suffix += 1

        new_entries[filename] = {
            "tmdb_id": "0",
            "title":   title,
            "slug":    slug,
        }
        logger.info("NEW  %-55s -> title: %-45s  slug: %s",
                    filename, title, slug)

    if not new_entries:
        logger.info("manifest.json is already up to date. No new files found.")
        return 0

    print(f"\n{len(new_entries)} new file(s) found.")

    if dry_run:
        print("Dry-run -- no changes written.")
        return len(new_entries)

    manifest.update(new_entries)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("manifest.json updated (%d total entries).", len(manifest))
    return len(new_entries)


def main():
    ap = argparse.ArgumentParser(
        description="Sync manifest.json with the screenplays directory."
    )
    ap.add_argument(
        "--screenplays-dir", default="../screenplays",
    )
    ap.add_argument(
        "--manifest", default="../screenplays/manifest.json",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without writing anything",
    )
    args = ap.parse_args()

    added = sync(
        screenplays_dir=args.screenplays_dir,
        manifest_path=args.manifest,
        dry_run=args.dry_run,
    )

    if added and not args.dry_run:
        print(
            "\nNext steps:\n"
            "  1. python tmdb_fetch.py   (resolve tmdb_ids + fetch metadata)\n"
            "  2. python convert.py      (convert new files to TXT)\n"
            "  3. python batch.py        (parse TXT -> scene JSON)\n"
        )


if __name__ == "__main__":
    main()
