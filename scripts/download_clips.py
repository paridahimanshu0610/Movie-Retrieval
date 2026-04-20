"""
download_clips.py

Downloads movie clips from YouTube based on timestamps defined in a JSON file.
- Skips clips that already exist on disk (safe to re-run after interruptions).
- Downloads only the segment between the given timestamps.
- Preserves good video + audio quality for use with VLMs and Whisper.

Requirements:
    pip install yt-dlp
    ffmpeg must be installed and on PATH (brew install ffmpeg on macOS)

Usage:
    # Use Chrome cookies (default) to bypass YouTube bot detection:
    python download_clips.py

    # Use a different browser:
    python download_clips.py --cookies-from-browser firefox
    python download_clips.py --cookies-from-browser safari
    python download_clips.py --cookies-from-browser edge
    python download_clips.py --cookies-from-browser brave

    # Use a manually exported cookies.txt file instead:
    python download_clips.py --cookies-file /path/to/cookies.txt

    # Process only one specific movie:
    python download_clips.py --movie "Harry Potter and the Goblet of Fire"

    # Override default paths:
    python download_clips.py --json /path/to/plot_summaries.json --base /path/to/project
"""

import json
import os
import subprocess
import argparse
import sys
from pathlib import Path


# ── Helpers ────────────────────────────────────────────────────────────────────

def ts_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS or MM:SS timestamp string to total seconds."""
    ts = ts.strip()
    parts = ts.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    else:
        return parts[0]


def parse_timestamp_range(timestamp: str):
    """
    Parse a timestamp range like '00:02:02 - 00:03:03'.
    Returns (start_seconds, duration_seconds).
    """
    parts = timestamp.split("-")
    if len(parts) != 2:
        raise ValueError(f"Cannot parse timestamp range: '{timestamp}'")
    start = ts_to_seconds(parts[0].strip())
    end   = ts_to_seconds(parts[1].strip())
    if end <= start:
        raise ValueError(f"End time must be after start time in: '{timestamp}'")
    return start, end - start


def strip_yt_timestamp(url: str) -> str:
    """Remove &t=... query param so yt-dlp fetches the full video."""
    from urllib.parse import urlparse, urlencode, parse_qs, urlunparse
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    params.pop("t", None)
    new_query = urlencode({k: v[0] for k, v in params.items()})
    return urlunparse(parsed._replace(query=new_query))


# ── Core download logic ────────────────────────────────────────────────────────

def download_clip(
    youtube_url: str,
    start_sec: float,
    duration_sec: float,
    output_path: Path,
    clip_description: str,
    cookie_args: list,
):
    """
    Download exactly [start_sec, start_sec + duration_sec] of a YouTube video
    using yt-dlp + ffmpeg, saving to output_path (.mp4).

    Strategy:
      1. yt-dlp fetches the best video+audio streams using browser cookies.
      2. ffmpeg trims to the exact timestamp range and re-encodes.

    Quality targets:
      - Video : h264, CRF 18 (visually lossless), <=1080p — great for VLMs
      - Audio : AAC 192 kbps — great for Whisper
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.mp4")
    raw_tmp  = output_path.with_suffix(".raw.mp4")

    clean_url = strip_yt_timestamp(youtube_url)

    print(f"\n  >> {clip_description}")
    print(f"     URL   : {clean_url}")
    print(f"     Range : {start_sec:.1f}s -> {start_sec + duration_sec:.1f}s  ({duration_sec:.1f}s)")
    print(f"     Output: {output_path}")

    # Step 1: download best quality mp4 (merged video + audio)
    yt_cmd = [
        "yt-dlp",
        "--format", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", str(raw_tmp),
        "--no-playlist",
        "--quiet",
        "--progress",
    ] + cookie_args + [clean_url]

    result = subprocess.run(yt_cmd)
    if result.returncode != 0:
        print(f"  FAILED (yt-dlp): {clip_description}")
        raw_tmp.unlink(missing_ok=True)
        return False

    # Step 2: trim to exact timestamp with ffmpeg
    ff_cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-ss", str(start_sec),        # input seek (fast)
        "-i", str(raw_tmp),
        "-t",  str(duration_sec),     # duration to keep
        "-c:v", "libx264",
        "-crf", "18",                 # visually lossless
        "-preset", "slow",            # better compression ratio
        "-pix_fmt", "yuv420p",        # broad VLM compatibility
        "-c:a", "aac",
        "-b:a", "192k",               # high quality for Whisper
        "-movflags", "+faststart",    # web-friendly mp4
        "-y",
        str(tmp_path),
    ]

    result = subprocess.run(ff_cmd)
    raw_tmp.unlink(missing_ok=True)   # always clean up raw file

    if result.returncode != 0:
        print(f"  FAILED (ffmpeg trim): {clip_description}")
        tmp_path.unlink(missing_ok=True)
        return False

    tmp_path.rename(output_path)
    size_mb = output_path.stat().st_size / 1_048_576
    print(f"  OK  ({size_mb:.1f} MB): {output_path.name}")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download movie clips from YouTube based on JSON metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json",
        default="/Users/himanshu/Documents/TAMU/Courses/ISR/Project/Movie Retrieval/data/plot_summaries.json",
        help="Path to the plot_summaries.json file",
    )
    parser.add_argument(
        "--base",
        default="/Users/himanshu/Documents/TAMU/Courses/ISR/Project/Movie Retrieval",
        help="Base directory of the project (clips/ folder will be created here)",
    )
    parser.add_argument(
        "--movie",
        default=None,
        help="(Optional) Only process this specific movie title",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default="chrome",
        dest="cookies_browser",
        metavar="BROWSER",
        help="Browser to pull cookies from: chrome, firefox, safari, edge, brave  [default: chrome]",
    )
    parser.add_argument(
        "--cookies-file",
        default=None,
        dest="cookies_file",
        metavar="FILE",
        help="(Alternative) Path to a Netscape-format cookies.txt exported from your browser",
    )
    args = parser.parse_args()

    json_path  = Path(args.json)
    base_dir   = Path(args.base)
    clips_root = base_dir / "clips"

    # ── Cookie auth ──────────────────────────────────────────────────────────
    if args.cookies_file:
        cookie_args = ["--cookies", args.cookies_file]
        print(f"[cookies] Using file: {args.cookies_file}")
    else:
        cookie_args = ["--cookies-from-browser", args.cookies_browser]
        print(f"[cookies] Using browser: {args.cookies_browser}")
        print(f"          Make sure {args.cookies_browser.title()} is open and you are logged into YouTube.")
        print(f"          Alternatively, export cookies manually and pass: --cookies-file cookies.txt\n")

    # ── Load JSON ────────────────────────────────────────────────────────────
    if not json_path.exists():
        print(f"ERROR: JSON file not found: {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        movies = list(data.values())
    elif isinstance(data, list):
        movies = data
    else:
        print("ERROR: Unexpected JSON structure (expected a dict or list at top level).")
        sys.exit(1)

    total_downloaded = 0
    total_skipped    = 0
    total_failed     = 0

    for movie in movies:
        title = movie.get("title", "Unknown")

        if args.movie and title != args.movie:
            continue

        clips = movie.get("clips", [])
        if not clips:
            continue

        print(f"\n{'=' * 60}")
        print(f"Movie : {title}  ({len(clips)} clip(s))")
        print(f"{'=' * 60}")

        movie_clip_dir = clips_root / title

        for clip in clips:
            description  = clip.get("description", "clip")
            timestamp    = clip.get("timestamp", "")
            filepath     = clip.get("filepath", "")
            youtube_link = clip.get("youtube_link", "")

            if not youtube_link:
                print(f"  [SKIP] No YouTube link for: {description}")
                continue
            if not timestamp:
                print(f"  [SKIP] No timestamp for: {description}")
                continue

            # Derive output filename from the JSON filepath field
            if filepath:
                filename = Path(filepath).name   # e.g. harry_picked_from_the_goblet.mp4
            else:
                filename = (
                    description.lower()
                    .replace(" ", "_")
                    .replace("/", "-")
                    + ".mp4"
                )

            output_path = movie_clip_dir / filename

            # Skip if already downloaded
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"  [EXISTS] Skipping: {filename}")
                total_skipped += 1
                continue

            try:
                start_sec, duration_sec = parse_timestamp_range(timestamp)
            except ValueError as e:
                print(f"  [ERROR] Timestamp parse failed for '{description}': {e}")
                total_failed += 1
                continue

            success = download_clip(
                youtube_url=youtube_link,
                start_sec=start_sec,
                duration_sec=duration_sec,
                output_path=output_path,
                clip_description=description,
                cookie_args=cookie_args,
            )
            if success:
                total_downloaded += 1
            else:
                total_failed += 1

    print(f"\n{'=' * 60}")
    print(f"Done.  Downloaded: {total_downloaded}  |  Skipped: {total_skipped}  |  Failed: {total_failed}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()