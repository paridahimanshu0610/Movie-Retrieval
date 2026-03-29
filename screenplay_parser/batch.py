"""
batch.py — Entry point. Reads manifest, iterates files, runs the full pipeline,
writes output JSON files.

Usage (raw files — runs ingest -> parse -> serialize):
    python batch.py --input-dir ./screenplays \\
                    --manifest ./screenplays/manifest.json \\
                    --output-dir ./output

Usage (pre-converted TXT files — skips ingest, runs parse -> serialize):
    python batch.py --converted-dir ./screenplays/converted \\
                    --manifest ./screenplays/manifest.json \\
                    --output-dir ./output
"""

import argparse
import json
import logging
import os
import sys

from ingest import ingest
from parse import parse
from serialize import serialize

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Parse screenplay files into structured JSON scene objects."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", help="Directory containing raw screenplay files (PDF/HTML/TXT).")
    source.add_argument("--converted-dir", help="Directory of pre-converted TXT files from convert.py.")
    parser.add_argument("--manifest",   required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    total = len(manifest)
    succeeded = 0
    failed = 0

    for filename, entry in manifest.items():
        tmdb_id = entry["tmdb_id"]
        title   = entry["title"]
        slug    = entry["slug"]
        output_path = os.path.join(args.output_dir, f"{slug}_scenes.json")

        if args.converted_dir:
            filepath = os.path.join(args.converted_dir, f"{slug}.txt")
            if not os.path.isfile(filepath):
                logger.warning("Converted file not found, skipping: %s", filepath)
                failed += 1
                continue
            try:
                with open(filepath, encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                logger.error("Read failed for %s: %s", filepath, e)
                _write_empty(output_path)
                failed += 1
                continue
        else:
            filepath = os.path.join(args.input_dir, filename)
            if not os.path.isfile(filepath):
                logger.warning("File not found, skipping: %s", filepath)
                failed += 1
                continue
            try:
                text = ingest(filepath)
            except ImportError:
                raise
            except Exception as e:
                logger.error("Ingest failed for %s: %s", filename, e)
                _write_empty(output_path)
                failed += 1
                continue

        try:
            raw_scenes = parse(text)
        except ImportError:
            raise
        except Exception as e:
            logger.error("Parse failed for %s: %s", filename, e)
            _write_empty(output_path)
            failed += 1
            continue

        if not raw_scenes:
            logger.warning("No scenes parsed for %s, writing empty output.", title)
            _write_empty(output_path)
            failed += 1
            continue

        try:
            scenes = serialize(raw_scenes, tmdb_id, slug)
        except Exception as e:
            logger.error("Serialize failed for %s: %s", filename, e)
            _write_empty(output_path)
            failed += 1
            continue

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(scenes, f, indent=2, ensure_ascii=False)

        logger.info("[OK] %s: %d scenes parsed", title, len(scenes))
        succeeded += 1

    print(f"\nDone. {total} files attempted -- {succeeded} succeeded, {failed} failed.")


def _write_empty(output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([], f)


if __name__ == "__main__":
    main()
