import json
from pathlib import Path

# ----------------------------
# Project directory
# ----------------------------
PROJECT_DIR = "/scratch/user/paridahimanshu0610/ISR"
PROJECT_ROOT = Path(PROJECT_DIR)

# ----------------------------
# Input paths
# ----------------------------
PLOT_SUMMARIES_PATH      = PROJECT_ROOT / "data"    / "plot_summaries.json"
AGGREGATED_CAPTIONS_PATH = PROJECT_ROOT / "outputs" / "aggregated_caption.json"
REFINED_CAPTIONS_PATH    = PROJECT_ROOT / "outputs" / "refined_caption.json"
TRANSCRIPTS_PATH         = PROJECT_ROOT / "outputs" / "clip_transcript.json"
OUTPUT_PATH              = PROJECT_ROOT / "outputs" / "final_clip_data.json"

# ----------------------------
# Validate inputs
# ----------------------------
for path in [PLOT_SUMMARIES_PATH, AGGREGATED_CAPTIONS_PATH, REFINED_CAPTIONS_PATH, TRANSCRIPTS_PATH]:
    assert path.exists(), f"Required file not found: {path}"

# ----------------------------
# Load all files
# ----------------------------
print("Loading input files...")

with open(PLOT_SUMMARIES_PATH, "r") as f:
    plot_summaries = json.load(f)

with open(AGGREGATED_CAPTIONS_PATH, "r") as f:
    all_aggregated_captions = json.load(f)

with open(REFINED_CAPTIONS_PATH, "r") as f:
    all_refined_captions = json.load(f)

with open(TRANSCRIPTS_PATH, "r") as f:
    all_transcripts = json.load(f)

print("All files loaded.\n")

# ----------------------------
# Helper: safely pull a nested key from a dict
# ----------------------------
def safe_get(source: dict, movie_key: str, clip_id: str, outer_key: str, inner_key: str):
    """
    Retrieves source[movie_key][clip_id][outer_key][inner_key].
    Returns None at each level if the key is missing, and prints a warning.
    """
    movie_data = source.get(movie_key)
    if movie_data is None:
        print(f"  [WARN] '{movie_key}' not found in source.")
        return None

    clip_data = movie_data.get(clip_id)
    if clip_data is None:
        print(f"  [WARN] '{clip_id}' not found under '{movie_key}' in source.")
        return None

    outer = clip_data.get(outer_key)
    if outer is None:
        print(f"  [WARN] '{outer_key}' not found for {movie_key}/{clip_id}.")
        return None

    value = outer.get(inner_key)
    if value is None:
        print(f"  [WARN] '{inner_key}' not found inside '{outer_key}' for {movie_key}/{clip_id}.")
    return value


def safe_get_direct(source: dict, movie_key: str, clip_id: str, key: str):
    """
    Retrieves source[movie_key][clip_id][key] (no outer_key nesting).
    Returns None if any level is missing, and prints a warning.
    """
    movie_data = source.get(movie_key)
    if movie_data is None:
        print(f"  [WARN] '{movie_key}' not found in source.")
        return None

    clip_data = movie_data.get(clip_id)
    if clip_data is None:
        print(f"  [WARN] '{clip_id}' not found under '{movie_key}' in source.")
        return None

    value = clip_data.get(key)
    if value is None:
        print(f"  [WARN] '{key}' not found for {movie_key}/{clip_id}.")
    return value


def build_concatenated_transcript(transcript: list) -> str:
    """Concatenate the 'text' field from each segment in the transcript list."""
    if not transcript:
        return ""
    return " ".join(
        entry.get("text", "").strip()
        for entry in transcript
        if entry.get("text", "").strip()
    )

# ----------------------------
# Build final structure
# ----------------------------
final_clip_data = {}

total_clips   = 0
merged_clips  = 0
missing_clips = 0

for movie_key, movie_data in plot_summaries.items():
    # Copy all original movie-level fields exactly as they are
    final_clip_data[movie_key] = {
        k: v for k, v in movie_data.items() if k != "clips"
    }
    final_clip_data[movie_key]["clips"] = {}

    clips = movie_data.get("clips", [])
    print(f"\n{'=' * 50}")
    print(f"Movie: {movie_key}  ({len(clips)} clips)")
    print(f"{'=' * 50}")

    for clip in clips:
        total_clips += 1
        clip_id = clip["clip_id"]
        print(f"\n  Clip: {clip_id}")

        # ------------------------------------------------------------------
        # From aggregated_caption.json → aggregated_caption → visual_description, subjects
        # ------------------------------------------------------------------
        visual_description = safe_get(
            all_aggregated_captions, movie_key, clip_id,
            outer_key="aggregated_caption", inner_key="visual_description"
        )
        subjects = safe_get(
            all_aggregated_captions, movie_key, clip_id,
            outer_key="aggregated_caption", inner_key="subjects"
        )

        # ------------------------------------------------------------------
        # From refined_caption.json → refined_caption → action, emotional_tone, thematic_connection
        # ------------------------------------------------------------------
        action = safe_get(
            all_refined_captions, movie_key, clip_id,
            outer_key="refined_caption", inner_key="action"
        )
        emotional_tone = safe_get(
            all_refined_captions, movie_key, clip_id,
            outer_key="refined_caption", inner_key="emotional_tone"
        )
        thematic_connection = safe_get(
            all_refined_captions, movie_key, clip_id,
            outer_key="refined_caption", inner_key="thematic_connection"
        )

        # ------------------------------------------------------------------
        # From clip_transcript.json → transcript, formatted_transcript
        # ------------------------------------------------------------------
        transcript = safe_get_direct(
            all_transcripts, movie_key, clip_id, key="transcript"
        )
        formatted_transcript = safe_get_direct(
            all_transcripts, movie_key, clip_id, key="formatted_transcript"
        )

        # ------------------------------------------------------------------
        # Build concatenated_transcript from the transcript segments
        # ------------------------------------------------------------------
        concatenated_transcript = build_concatenated_transcript(transcript or [])

        # ------------------------------------------------------------------
        # Assemble the clip entry:
        #   - all original clip fields from plot_summaries (clip_id, filepath, etc.)
        #   - additional fields from the pipeline outputs
        # ------------------------------------------------------------------
        final_clip_data[movie_key]["clips"][clip_id] = {
            **clip,                                         # all original clip metadata
            "visual_description":     visual_description,
            "subjects":               subjects,
            "action":                 action,
            "emotional_tone":         emotional_tone,
            "thematic_connection":    thematic_connection,
            "transcript":             transcript,
            "formatted_transcript":   formatted_transcript,
            "concatenated_transcript":concatenated_transcript,
        }

        any_missing = any(v is None for v in [
            visual_description, subjects, action, emotional_tone,
            thematic_connection, transcript, formatted_transcript
        ])
        if any_missing:
            missing_clips += 1
            print(f"    [WARN] Some fields are missing — see warnings above.")
        else:
            merged_clips += 1
            print(f"    [OK] All fields merged successfully.")

# ----------------------------
# Save output
# ----------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(final_clip_data, f, indent=2)

# ----------------------------
# Summary
# ----------------------------
print(f"\n{'=' * 60}")
print("MERGE COMPLETE")
print(f"{'=' * 60}")
print(f"  Total clips          : {total_clips}")
print(f"  Fully merged         : {merged_clips}")
print(f"  Partially merged     : {missing_clips}  (check WARN lines above)")
print(f"\n  Output saved to: {OUTPUT_PATH}")