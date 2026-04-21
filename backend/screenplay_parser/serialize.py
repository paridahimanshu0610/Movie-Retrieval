"""
serialize.py — Take raw scene dicts from parse.py plus manifest metadata,
produce final scene objects matching the output schema.
"""

import re

_ABBREVS = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Jr|Sr|Lt|Sgt|Cpl|Pvt|St|Vs|No|Vol|Dept|Ext|Int)\."
    r"\s+[A-Z]",
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Lines with fewer than 3 alphabetic characters are OCR artifacts —
# page numbers, scan noise, stray symbols (e.g. ":2.", "/3", "I~", "'f5").
# Valid short lines like "BAM!" or "BANG." have at least 3 letters.
_MIN_ALPHA_CHARS = 3

def _is_ocr_garbage(text: str) -> bool:
    return sum(1 for c in text if c.isalpha()) < _MIN_ALPHA_CHARS


def _split_action_block(block: str) -> list:
    """
    Unwrap soft-wrapped lines then split on sentence-ending punctuation,
    protecting known abbreviations from false splits.
    """
    text = re.sub(r"\s*\n\s*", " ", block).strip()
    if not text:
        return []

    protected = _ABBREVS.sub(lambda m: m.group().replace(". ", ".\x00"), text)
    parts = _SENT_SPLIT_RE.split(protected)
    return [p.strip().replace("\x00", " ") for p in parts if p.strip()]


def serialize(raw_scenes: list, tmdb_id: str, slug: str) -> list:
    """Returns list of scene objects matching the output schema."""
    output = []
    scene_number = 0

    for raw in raw_scenes:
        heading = raw.get("heading", "")
        if not heading:
            continue

        scene_number += 1

        action_lines = []
        for block in raw.get("action_lines", []):
            if block and block.strip():
                for line in _split_action_block(block):
                    if not _is_ocr_garbage(line):
                        action_lines.append(line)

        dialogue = [
            {
                "character": d["character"],
                "line": d["line"],
                "parenthetical": d.get("parenthetical"),
            }
            for d in raw.get("dialogue", [])
            if d.get("character")
        ]

        time_of_day = raw.get("time_of_day", "") or "UNSPECIFIED"

        seen = set()
        characters = []
        for d in dialogue:
            c = d["character"]
            if c not in seen:
                seen.add(c)
                characters.append(c)

        action_words = sum(len(a.split()) for a in action_lines)
        dialogue_words = sum(len(d["line"].split()) for d in dialogue)
        word_count = action_words + dialogue_words

        scene = {
            "scene_id": f"{slug}_scene_{scene_number:03d}",
            "movie_id": tmdb_id,
            "scene_number": scene_number,
            "heading": heading,
            "int_ext": raw.get("int_ext", "INT"),
            "location": raw.get("location", ""),
            "time_of_day": time_of_day,
            "characters": characters,
            "word_count": word_count,
            "action_lines": action_lines,
            "dialogue": dialogue,
        }
        output.append(scene)

    return output
