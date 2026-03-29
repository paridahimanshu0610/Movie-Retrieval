"""
test_ingest.py — Smoke test for the full ingest → parse → serialize pipeline.

Creates a small in-memory screenplay (3 scenes), writes it to a temp .txt file,
then exercises the full pipeline and asserts basic correctness.
"""

import json
import os
import sys
import tempfile

# Allow running from the screenplay_parser directory
sys.path.insert(0, os.path.dirname(__file__))

from ingest import ingest
from parse import parse
from serialize import serialize

# ---------------------------------------------------------------------------
# Sample screenplay — standard format
# ---------------------------------------------------------------------------


# Proper Hollywood Standard indentation for ScreenPy v2:
#   Character name: 25 spaces (>=20, triggers center_indent detection)
#   Dialogue lines: 10 spaces (10-19, triggers dialogue_indent but NOT center_indent)
#   Parentheticals: 15 spaces (any indent, detected by "^\s*\([^)]+\)\s*$" pattern)
#   Action lines:    0 spaces (not indented, treated as stage direction)
SAMPLE_SCREENPLAY = (
    "FADE IN:\n"
    "\n"
    "INT. COFFEE SHOP - DAY\n"
    "\n"
    "The shop is bustling. A BARISTA wipes down the counter.\n"
    "\n"
    "                         BARISTA\n"
    "          Morning rush. Never ends.\n"
    "\n"
    "NEO enters, disheveled, clutching a laptop.\n"
    "\n"
    "                         NEO\n"
    "          One black coffee.\n"
    "\n"
    "                         BARISTA\n"
    "               (skeptical)\n"
    "          Name?\n"
    "\n"
    "                         NEO\n"
    "          ...Neo.\n"
    "\n"
    "EXT. CITY STREET - NIGHT\n"
    "\n"
    "Rain hammers the pavement. A PAY PHONE rings in the distance.\n"
    "\n"
    "Neo steps outside, collar up against the cold.\n"
    "\n"
    "                         NEO\n"
    "          I know you're out there.\n"
    "\n"
    "INT. ABANDONED WAREHOUSE - NIGHT\n"
    "\n"
    "Bare concrete. A single bulb swings overhead.\n"
    "\n"
    "Morpheus stands at the center of the room, arms folded.\n"
    "\n"
    "                         MORPHEUS\n"
    "          You're late.\n"
    "\n"
    "                         NEO\n"
    "          Traffic.\n"
    "\n"
    "                         MORPHEUS\n"
    "               (quietly)\n"
    "          There is no traffic. Not anymore.\n"
    "\n"
    "Action resumes as Morpheus walks to the window.\n"
    "\n"
    "FADE OUT.\n"
)

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_pipeline():
    # Write sample screenplay to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(SAMPLE_SCREENPLAY)
        tmp_path = tmp.name

    try:
        # 1. Ingest
        text = ingest(tmp_path)
        assert isinstance(text, str), "ingest() must return a string"
        assert len(text) > 0, "ingest() returned empty string"
        print(f"[ingest] OK — {len(text)} chars extracted")

        # 2. Parse
        raw_scenes = parse(text)
        assert isinstance(raw_scenes, list), "parse() must return a list"
        print(f"[parse]  OK — {len(raw_scenes)} raw scene(s) found")

        # 3. Serialize
        scenes = serialize(raw_scenes, tmdb_id="tt9999999", slug="test_movie")
        assert isinstance(scenes, list), "serialize() must return a list"

        # Core assertions
        assert len(scenes) == 3, (
            f"Expected 3 scenes, got {len(scenes)}. "
            "Make sure all three INT/EXT headings were detected."
        )

        for scene in scenes:
            # scene_id format
            assert scene["scene_id"].startswith("test_movie_scene_"), (
                f"Bad scene_id: {scene['scene_id']}"
            )
            # Zero-padded 3-digit scene number
            num_part = scene["scene_id"].split("_scene_")[1]
            assert len(num_part) == 3 and num_part.isdigit(), (
                f"scene_id number not zero-padded to 3 digits: {scene['scene_id']}"
            )
            # action_lines is a list
            assert isinstance(scene["action_lines"], list), (
                f"action_lines must be a list, got {type(scene['action_lines'])}"
            )
            # dialogue entries have required keys
            for d in scene["dialogue"]:
                assert "character" in d, "dialogue entry missing 'character'"
                assert "line" in d, "dialogue entry missing 'line'"

        print(f"[serialize] OK — {len(scenes)} scenes serialized")
        print()
        print("=== First scene (visual verify) ===")
        print(json.dumps(scenes[0], indent=2))

        if len(scenes) > 1:
            print()
            print("=== Second scene ===")
            print(json.dumps(scenes[1], indent=2))

        if len(scenes) > 2:
            print()
            print("=== Third scene ===")
            print(json.dumps(scenes[2], indent=2))

        print()
        print("All assertions passed.")

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    test_pipeline()
