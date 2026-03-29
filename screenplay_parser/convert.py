"""
convert.py — Convert screenplay files (PDF, HTML, TXT) to clean plain-text.

Produces a normalized .txt file with canonical Hollywood Standard indentation
that downstream parsers can process reliably:

    Scene headings  :  0 sp  (left margin)
    Action lines    :  0 sp  (left margin)
    Character names : 25 sp
    Dialogue        : 10 sp
    Parentheticals  : 15 sp

Usage (single file):
    python convert.py input.pdf [output.txt]

Usage (batch via manifest):
    python convert.py --input-dir ./screenplays \\
                      --manifest  ./screenplays/manifest.json \\
                      --output-dir ./screenplays/converted
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                    stream=sys.stderr)
logger = logging.getLogger(__name__)

COL_ACTION     =  0
COL_DIALOGUE   = 10
COL_PAREN      = 15
COL_CHARACTER  = 25


def convert(filepath: str) -> str:
    """
    Convert a screenplay file (PDF / HTML / TXT) to clean plain-text.
    Returns the converted text as a string.
    """
    ext = Path(filepath).suffix.lower()

    if ext == ".pdf":
        raw = _extract_pdf(filepath)
    elif ext in (".htm", ".html"):
        raw = _extract_html(filepath)
    else:
        raw = _extract_txt(filepath)

    raw = _fix_mojibake(raw)
    raw = _fix_doubled_chars(raw)
    raw = _strip_revision_stars(raw)
    raw = _strip_noise(raw)
    raw = _strip_scene_numbers(raw)
    raw = _normalize_columns(raw)
    raw = _final_cleanup(raw)
    return raw


def _extract_pdf(filepath: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber required: pip install pdfplumber")

    pages = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True)
            if text:
                pages.append(text)
    return "\n".join(pages)


def _extract_html(filepath: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 required: pip install beautifulsoup4")

    import chardet
    raw_bytes = Path(filepath).read_bytes()
    enc = chardet.detect(raw_bytes).get("encoding") or "utf-8"
    try:
        html = raw_bytes.decode(enc)
    except (UnicodeDecodeError, LookupError):
        html = raw_bytes.decode("utf-8", errors="replace")

    soup = BeautifulSoup(html, "html.parser")

    # <pre> tags preserve indentation — prefer them over raw body text
    pre_tags = soup.find_all("pre")
    if pre_tags:
        return "\n".join(t.get_text() for t in pre_tags)

    container = soup.find("body") or soup
    for br in container.find_all("br"):
        br.replace_with("\n")
    text = container.get_text()
    text = text.replace("\xa0", " ").replace("&nbsp;", " ")

    lines = [l for l in text.splitlines() if l.strip()]
    if lines and sum(len(l) for l in lines) / len(lines) < 20:
        raise ValueError("HTML structure unrecognisable: average line too short.")
    return text


def _extract_txt(filepath: str) -> str:
    import chardet
    raw_bytes = Path(filepath).read_bytes()
    result = chardet.detect(raw_bytes)
    enc = result.get("encoding")
    conf = result.get("confidence", 0)
    if not enc or conf < 0.5:
        enc = "utf-8"
    try:
        return raw_bytes.decode(enc)
    except (UnicodeDecodeError, LookupError):
        return raw_bytes.decode("utf-8", errors="replace")


# pdfplumber may produce typographic punctuation that won't match plain-keyboard
# search queries, so normalize it all to ASCII equivalents.
_UNICODE_PUNCT = str.maketrans({
    "\ufffd": "'",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "--",
    "\u2026": "...",
})

def _fix_mojibake(text: str) -> str:
    return text.translate(_UNICODE_PUNCT)


# Some PDFs render two overlapping text layers (e.g. revision watermarks),
# producing "SSaallmmoonn" instead of "Salmon". Detect and de-duplicate.
def _fix_doubled_chars(text: str) -> str:
    result = []
    for line in text.splitlines():
        result.append(_dedouble_line(line) if _is_doubled_line(line) else line)
    return "\n".join(result)


def _is_doubled_line(line: str) -> bool:
    chars = [c for c in line if not c.isspace()]
    if len(chars) < 8 or len(chars) % 2 != 0:
        return False
    matching = sum(1 for i in range(0, len(chars), 2) if chars[i] == chars[i + 1])
    return matching / (len(chars) // 2) >= 0.80


def _dedouble_line(line: str) -> str:
    """
    De-double a line where each non-space character appears twice.
    Spaces are not doubled, so advance by 1 for spaces and 2 for other chars.
    """
    leading = len(line) - len(line.lstrip(" "))
    content = line[leading:]
    result = []
    i = 0
    while i < len(content):
        c = content[i]
        if c == " ":
            result.append(c)
            i += 1
        else:
            result.append(c)
            i += 2
    return " " * leading + "".join(result)


# Some shooting scripts place a '*' in the far right margin on revised lines.
_TRAILING_STAR_RE = re.compile(r"\s+\*\s*$")

def _strip_revision_stars(text: str) -> str:
    return "\n".join(
        _TRAILING_STAR_RE.sub("", line) for line in text.splitlines()
    )


_PAGE_NUMBER_RE  = re.compile(r"^\s*\d+\.?\s*$")
_HEADER_WITH_NUM = re.compile(r"^(.*?)\s+\d+\.?\s*$")
_CONTINUED_RE    = re.compile(r"^\s*\(?CONTINUED(?::\s*\(\d+\))?\)?\s*$", re.IGNORECASE)


def _header_base(stripped: str) -> str | None:
    m = _HEADER_WITH_NUM.match(stripped)
    return m.group(1).strip() if m else None


def _strip_noise(text: str) -> str:
    lines = text.splitlines()

    base_counts: Counter = Counter()
    for line in lines:
        s = line.strip()
        if not s or _PAGE_NUMBER_RE.match(s):
            continue
        base = _header_base(s)
        if base:
            base_counts[base] += 1

    # Bases appearing >10 times are repeated page headers/footers
    repeated = {b for b, n in base_counts.items() if n > 10}

    cleaned = []
    for line in lines:
        s = line.strip()
        if not s:
            cleaned.append(line)
            continue
        if _PAGE_NUMBER_RE.match(s):
            continue
        if _CONTINUED_RE.match(s):
            continue
        base = _header_base(s)
        if base and base in repeated:
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


# Shooting scripts embed scene numbers on both sides of headings:
#   "       1       EXT. PARKING LOT - EARLY MORNING                  1"
# Also handles alphanumeric numbers: "2D   INT. HALL - DAY   2D"
_SCENE_NUM_HEADING_RE = re.compile(
    r"^(\s*)[A-Z0-9]+\s+"
    r"((?:INT\.?(?:/EXT\.?)?|EXT\.?(?:/INT\.?)?|I/E)"
    r"(?:\s+.+?)?)"
    r"\s+[A-Z0-9]+\s*$",
    re.IGNORECASE,
)


def _strip_scene_numbers(text: str) -> str:
    result = []
    for line in text.splitlines():
        m = _SCENE_NUM_HEADING_RE.match(line)
        result.append(m.group(1) + m.group(2).strip() if m else line)
    return "\n".join(result)


_CANONICAL_TARGETS = [COL_ACTION, COL_DIALOGUE, COL_PAREN, COL_CHARACTER]


def _normalize_columns(text: str) -> str:
    """
    Remap screenplay indentation to canonical columns.

    Strategy: build a histogram of leading-space counts, identify the highest
    significant indent level as character names (-> COL_CHARACTER=25), levels
    above char_level/2 as dialogue/parenthetical (-> COL_DIALOGUE=10), and
    levels at or below char_level/2 as action (-> COL_ACTION=0).

    No-ops when the file is already at or below the canonical character column.
    """
    lines = text.splitlines()

    counts: Counter = Counter()
    for line in lines:
        if line.strip():
            n = len(line) - len(line.lstrip(" "))
            if n > 0:
                counts[n] += 1

    if not counts:
        return text

    total = sum(counts.values())
    significant = sorted(k for k, v in counts.items() if v >= total * 0.02)

    if len(significant) < 2:
        return text

    char_level = significant[-1]

    if char_level <= COL_CHARACTER:
        return text

    threshold = char_level / 2

    remap = {}
    for level in significant:
        if level == char_level:
            remap[level] = COL_CHARACTER
        elif level > threshold:
            remap[level] = COL_DIALOGUE
        else:
            remap[level] = COL_ACTION

    def nearest(n: int) -> int:
        return remap[min(significant, key=lambda s: abs(s - n))]

    result = []
    for line in lines:
        if not line.strip():
            result.append(line)
            continue
        n = len(line) - len(line.lstrip(" "))
        if n == 0:
            result.append(line)
            continue
        result.append(" " * remap.get(n, nearest(n)) + line.lstrip(" "))

    return "\n".join(result)


def _final_cleanup(text: str) -> str:
    """Strip trailing whitespace and collapse runs of 3+ blank lines to 2."""
    lines = [line.rstrip() for line in text.splitlines()]
    result = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                result.append("")
        else:
            blank_run = 0
            result.append(line)
    return "\n".join(result)


def _convert_file(src: Path, dst: Path) -> None:
    try:
        text = convert(str(src))
    except Exception as e:
        logger.error("FAILED %s: %s", src.name, e)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    lines = text.count("\n") + 1
    logger.info("[OK] %s -> %s  (%d lines)", src.name, dst.name, lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert screenplay PDF/HTML/TXT to clean TXT.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("input", nargs="?", help="Single input file")
    group.add_argument("--input-dir", help="Directory of screenplay files (requires --manifest)")
    ap.add_argument("output", nargs="?", help="Output .txt path (single-file mode)")
    ap.add_argument("--manifest", help="Path to manifest.json (batch mode)")
    ap.add_argument("--output-dir", default="./converted", help="Output directory (batch mode)")
    args = ap.parse_args()

    if args.input:
        src = Path(args.input)
        dst = Path(args.output) if args.output else src.with_suffix(".txt")
        if dst == src:
            dst = src.with_name(src.stem + "_converted.txt")
        _convert_file(src, dst)
        return

    if not args.manifest:
        ap.error("--manifest is required with --input-dir")

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir)

    ok = fail = 0
    for filename, entry in manifest.items():
        slug = entry.get("slug", Path(filename).stem)
        src = Path(args.input_dir) / filename
        dst = out_dir / f"{slug}.txt"
        if not src.exists():
            logger.warning("NOT FOUND: %s", src)
            fail += 1
            continue
        _convert_file(src, dst)
        ok += 1

    print(f"\nDone. {ok + fail} files -- {ok} converted, {fail} failed.")


if __name__ == "__main__":
    main()
