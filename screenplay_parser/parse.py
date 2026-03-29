"""
parse.py — Take clean text string, run ScreenPy parser, return list of raw scene dicts.

Two-phase approach:
  Phase 1 — ScreenPy detects INT/EXT master scene headings and their line positions.
  Phase 2 — Our own regex parser walks the text lines within each scene to extract
             action_lines and dialogue, bypassing ScreenPy's over-eager
             _parse_stage_direction which bundles dialogue into heading content.
"""

import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

# Requires INT/EXT followed by a non-word char (prevents "INTERSTELLAR" matching INT)
_HEADING_RE = re.compile(r"^\s*(INT\.?(?:/EXT\.?)?|EXT\.?|I/E)\b", re.IGNORECASE)

_SHOT_KEYWORDS = frozenset([
    "SHOT", "ANGLE", "CUT", "FADE", "DISSOLVE", "INSERT", "POV", "P.O.V",
    "CLOSE", "CLOSEUP", "CLOSE-UP", "CU", "ECU", "WIDE", "WS", "EWS",
    "ZOOM", "TILT", "PAN", "CRANE", "TRACKING", "DOLLY", "STEADICAM",
    "ESTABLISHING", "EST", "AERIAL", "UNDERWATER", "HANDHELD", "MOVING",
    "MEDIUM", "MED", "MS", "FULL", "LONG", "TWO", "THREE", "REVERSE",
])

_CHAR_SUFFIX_RE = re.compile(r"\s*\(.*?\)\s*$")
_PAREN_RE = re.compile(r"^\s*\([^)]+\)\s*$")

_CENTER_INDENT = 20
_DIALOGUE_INDENT = 10


def parse(text: str) -> list:
    """
    Returns list of raw scene dicts with keys:
      heading, int_ext, location, time_of_day, action_lines, dialogue
    dialogue items have keys: character, line, parenthetical
    """
    try:
        from screenpy import ScreenplayParser
        from screenpy.models import LocationType
    except ImportError as e:
        raise ImportError(
            "ScreenPy not installed. Run: "
            "pip install git+https://github.com/drwiner/ScreenPy.git"
        ) from e

    try:
        screenplay = ScreenplayParser().parse(text)
    except Exception as e:
        logger.warning("ScreenPy parse() failed: %s", e)
        return []

    text_lines = text.split("\n")
    master_segs = [seg for seg in screenplay.segments if seg.is_master_segment]

    # Global fallback indent (used for short scenes that lack enough character lines
    # to self-detect their own indentation level).
    global_char_indent = _detect_char_indent(text_lines)

    scenes = []

    for idx, seg in enumerate(master_segs):
        heading_obj = seg.heading
        raw_heading = heading_obj.raw_text.strip()

        if not _HEADING_RE.match(raw_heading):
            continue

        int_ext = _normalize_location_type(heading_obj.location_type, LocationType)
        location = " - ".join(heading_obj.locations) if heading_obj.locations else ""
        time_of_day = heading_obj.time_of_day or "UNSPECIFIED"

        scene_start = seg.start_pos + 1
        if idx + 1 < len(master_segs):
            scene_end = master_segs[idx + 1].start_pos
        else:
            scene_end = len(text_lines)

        scene_lines = text_lines[scene_start:scene_end]

        # Per-scene indent detection handles mixed-format screenplays (e.g. a
        # screenplay where early scenes use 10-space character indentation and
        # later scenes use 25-space).  Fall back to the global value for scenes
        # that are too short to produce a reliable measurement.
        n_char_candidates = sum(
            1 for l in scene_lines if l.strip() and _is_character_name(l.strip())
        )
        char_indent = (
            _detect_char_indent(scene_lines)
            if n_char_candidates >= 3
            else global_char_indent
        )

        action_lines, dialogue = _parse_scene_content(scene_lines, char_indent)

        scenes.append({
            "heading": raw_heading,
            "int_ext": int_ext,
            "location": location,
            "time_of_day": time_of_day,
            "action_lines": action_lines,
            "dialogue": dialogue,
        })

    return scenes


def _detect_char_indent(lines: list) -> int:
    """
    Detect the indentation level used for character names in this file.

    Standard format uses >=20sp. Some PDFs (e.g. No Hard Feelings, The Beekeeper)
    use 10sp or 0sp. We count lines that pass _is_character_name() at each indent
    level and return the dominant level. If >=70% are at >=_CENTER_INDENT, standard.
    """
    counts: Counter = Counter()
    for line in lines:
        s = line.strip()
        if s and _is_character_name(s):
            n = len(line) - len(line.lstrip(" "))
            counts[n] += 1
    if not counts:
        return _CENTER_INDENT
    standard = sum(v for k, v in counts.items() if k >= _CENTER_INDENT)
    total = sum(counts.values())
    if total == 0 or standard / total >= 0.7:
        return _CENTER_INDENT
    return counts.most_common(1)[0][0]


def _parse_scene_content(lines: list, char_indent: int = _CENTER_INDENT) -> tuple:
    """
    Walk the text lines of a single scene and separate action blocks from dialogue.

    Standard format (char_indent >= _CENTER_INDENT = 20):
      >= 20 spaces  ->  character name
      10-19 spaces  ->  dialogue line (when in dialogue state)
       0-9 spaces   ->  action line

    Non-standard format (char_indent < _CENTER_INDENT, e.g. 10sp or 0sp):
      char_indent spaces + all-caps short line -> character name
      any non-blank line after character name  -> dialogue (until blank line)
      blank line -> ends dialogue block, returns to action state
    """
    non_standard = char_indent < _CENTER_INDENT

    action_lines = []
    dialogue = []

    action_buf = []
    char_name = None
    paren = None
    dlg_buf = []
    state = "action"

    def flush_action():
        if action_buf:
            text = "\n".join(action_buf).strip()
            if text:
                action_lines.append(text)
            action_buf.clear()

    def flush_dialogue():
        nonlocal char_name, paren
        if char_name and dlg_buf:
            dialogue.append({
                "character": char_name,
                "line": " ".join(dlg_buf).strip(),
                "parenthetical": paren,
            })
            dlg_buf.clear()
        paren = None

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if state == "action" and action_buf:
                flush_action()
            elif state == "dialogue" and non_standard:
                # In non-standard files, blank lines delimit speeches
                flush_dialogue()
                char_name = None
                state = "action"
            continue

        indent = len(line) - len(line.lstrip(" "))

        if indent >= _CENTER_INDENT:
            if _is_character_name(stripped):
                flush_action()
                flush_dialogue()
                char_name = _CHAR_SUFFIX_RE.sub("", stripped).strip() or None
                state = "dialogue"
            else:
                flush_dialogue()
                char_name = None
                state = "action"
                action_buf.append(stripped)

        elif non_standard and indent >= char_indent and _is_character_name(stripped):
            flush_action()
            flush_dialogue()
            char_name = _CHAR_SUFFIX_RE.sub("", stripped).strip() or None
            state = "dialogue"

        elif _PAREN_RE.match(line) and state == "dialogue":
            paren = stripped.strip("()")

        elif indent >= _DIALOGUE_INDENT and state == "dialogue" and not non_standard:
            dlg_buf.append(stripped)

        elif state == "dialogue" and non_standard:
            dlg_buf.append(stripped)

        else:
            if state == "dialogue":
                if dlg_buf:
                    flush_dialogue()
                state = "action"
            action_buf.append(stripped)

    flush_action()
    flush_dialogue()

    return action_lines, dialogue


def _normalize_location_type(location_type, LocationType) -> str:
    if location_type == LocationType.INTERIOR:
        return "INT"
    elif location_type == LocationType.EXTERIOR:
        return "EXT"
    elif location_type == LocationType.INT_EXT:
        return "INT/EXT"
    return "INT"


def _is_character_name(text: str) -> bool:
    """
    Return True if text looks like a screenplay character name:
      - All uppercase (plus spaces, apostrophes, periods, hyphens, #, digits)
      - 4 words or fewer
      - Not a scene heading (INT./EXT.)
      - No shot-type keywords (CUT TO, FADE OUT, etc.)
    """
    if not text:
        return False
    if _HEADING_RE.match(text):
        return False
    base = _CHAR_SUFFIX_RE.sub("", text).strip()
    if not base:
        return False
    if not re.match(r"^[A-Z][A-Z0-9#\s'./-]*$", base):
        return False
    if len(base.split()) > 4:
        return False
    if set(base.upper().split()) & _SHOT_KEYWORDS:
        return False
    return True
