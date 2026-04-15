"""
ingest.py — Accept a file path, detect format, extract clean text with
indentation preserved, strip noise (repeated headers/footers, page numbers).
"""

import re
from collections import Counter


def ingest(filepath: str) -> str:
    """
    Returns clean screenplay text as a single string.
    Raises ValueError if format is unsupported or HTML structure is unrecognizable.
    """
    ext = _get_extension(filepath)

    if ext == ".pdf":
        text = _extract_pdf(filepath)
    elif ext in (".htm", ".html"):
        text = _extract_html(filepath)
    else:
        text = _extract_txt(filepath)

    text = _strip_noise(text)
    text = _strip_scene_numbers(text)
    return _normalize_indentation(text)


def _get_extension(filepath: str) -> str:
    import os
    _, ext = os.path.splitext(filepath)
    return ext.lower()


def _extract_pdf(filepath: str) -> str:
    import pdfplumber

    pages = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(layout=True)
            if page_text is not None:
                pages.append(page_text)
    return "\n".join(pages)


def _extract_txt(filepath: str) -> str:
    import chardet

    with open(filepath, "rb") as f:
        raw = f.read()

    result = chardet.detect(raw)
    encoding = result.get("encoding")
    confidence = result.get("confidence", 0)

    if not encoding or confidence < 0.5:
        encoding = "utf-8"

    try:
        return raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return raw.decode("utf-8", errors="replace")


def _extract_html(filepath: str) -> str:
    from bs4 import BeautifulSoup

    with open(filepath, "rb") as f:
        raw = f.read()

    import chardet
    result = chardet.detect(raw)
    encoding = result.get("encoding") or "utf-8"
    try:
        content = raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        content = raw.decode("utf-8", errors="replace")

    soup = BeautifulSoup(content, "html.parser")

    # <pre> tags preserve indentation — prefer them over raw body text
    pre_tags = soup.find_all("pre")
    if pre_tags:
        text = "\n".join(tag.get_text() for tag in pre_tags)
    else:
        container = soup.find("body") or soup
        for br in container.find_all("br"):
            br.replace_with("\n")
        text = container.get_text()
        text = text.replace("\xa0", " ").replace("&nbsp;", " ")

    lines = [l for l in text.splitlines() if l.strip()]
    if lines:
        avg_len = sum(len(l) for l in lines) / len(lines)
        if avg_len < 20:
            raise ValueError(
                f"HTML structure unrecognizable: average line length {avg_len:.1f} < 20."
            )

    return text


# Matches standalone page number lines, e.g. "47" or "47."
_PAGE_NUMBER_RE = re.compile(r"^\s*\d+\.?\s*$")

# Matches lines ending with a trailing page number, e.g. "THE MATRIX  47"
_HEADER_FOOTER_RE = re.compile(r"^(.*?)\s+\d+\s*$")

_CONTINUED_RE = re.compile(r"^\s*(CONTINUED:|CONT'D)\s*$", re.IGNORECASE)


def _normalize_base(line: str) -> str:
    stripped = line.strip()
    m = _HEADER_FOOTER_RE.match(stripped)
    if m:
        return m.group(1).strip()
    return stripped


def _strip_noise(text: str) -> str:
    lines = text.splitlines()

    # Count bases only for lines that actually end with a page number.
    # Pure words like "BRUCE" must not be counted as they're character names.
    base_counts: Counter = Counter()
    for line in lines:
        stripped = line.strip()
        if not stripped or _PAGE_NUMBER_RE.match(stripped):
            continue
        if _HEADER_FOOTER_RE.match(stripped):
            base = _normalize_base(stripped)
            if base:
                base_counts[base] += 1

    repeated_bases = {base for base, count in base_counts.items() if count > 10}

    cleaned = []
    for line in lines:
        stripped = line.strip()

        if _PAGE_NUMBER_RE.match(stripped):
            continue
        if _CONTINUED_RE.match(stripped):
            continue
        if stripped:
            base = _normalize_base(stripped)
            if base in repeated_bases:
                continue

        cleaned.append(line)

    return "\n".join(cleaned)


# Shooting scripts embed scene numbers on both sides of each heading:
#   "       1       EXT. PARKING LOT - EARLY MORNING                         1"
_SCENE_NUMBERED_HEADING_RE = re.compile(
    r"^(\s*)\d+\s+"
    r"((?:INT\.?(?:/EXT\.?)?|EXT\.?|I/E)"
    r"(?:\s+.+?)?)"
    r"\s+\d+\s*$",
    re.IGNORECASE,
)


def _strip_scene_numbers(text: str) -> str:
    result = []
    for line in text.splitlines():
        m = _SCENE_NUMBERED_HEADING_RE.match(line)
        if m:
            leading_ws = m.group(1)
            heading = m.group(2).strip()
            result.append(leading_ws + heading)
        else:
            result.append(line)
    return "\n".join(result)


# Target column positions for the four screenplay element levels
_INDENT_TARGETS = [0, 12, 15, 25, 45]


def _normalize_indentation(text: str) -> str:
    """
    Detect the screenplay's indentation column structure and remap to
    ScreenPy-compatible levels when necessary.

    Only remaps if the second-highest significant indent level is >=20 spaces,
    meaning dialogue lines would collide with ScreenPy's center_indent threshold.
    No-ops if the file is already correctly indented.
    """
    lines = text.splitlines()

    indent_counts: Counter = Counter()
    for line in lines:
        if line.strip():
            n = len(line) - len(line.lstrip(" "))
            if n > 0:
                indent_counts[n] += 1

    if not indent_counts:
        return text

    total_indented = sum(indent_counts.values())

    # Filter out rare one-off indents (titles, transitions) below 2% frequency
    significant = sorted(
        k for k, v in indent_counts.items()
        if v >= total_indented * 0.02
    )

    if len(significant) < 2:
        return text

    dialogue_level = significant[-2]
    if dialogue_level < 20:
        return text

    remap = {
        level: (_INDENT_TARGETS[i] if i < len(_INDENT_TARGETS) else _INDENT_TARGETS[-1])
        for i, level in enumerate(significant)
    }

    def _nearest_target(n: int) -> int:
        nearest = min(significant, key=lambda s: abs(s - n))
        return remap[nearest]

    result = []
    for line in lines:
        if not line.strip():
            result.append(line)
            continue
        n = len(line) - len(line.lstrip(" "))
        if n == 0:
            result.append(line)
            continue
        new_indent = remap.get(n, _nearest_target(n))
        result.append(" " * new_indent + line.lstrip(" "))

    return "\n".join(result)
