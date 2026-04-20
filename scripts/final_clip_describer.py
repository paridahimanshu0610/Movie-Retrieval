import os

from dotenv import load_dotenv

# ----------------------------
# Project directory (Grace)
# ----------------------------
PROJECT_DIR = "/scratch/user/paridahimanshu0610/ISR"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = f"{PROJECT_DIR}/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = f"{PROJECT_DIR}/hf_cache"
# Load variables from .env into environment
load_dotenv()

import json
import datetime
import torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(PROJECT_DIR)

# ----------------------------
# Paths
# ----------------------------
PLOT_SUMMARIES_PATH      = PROJECT_ROOT / "data"    / "plot_summaries.json"
AGGREGATED_CAPTIONS_PATH = PROJECT_ROOT / "outputs" / "aggregated_caption.json"
TRANSCRIPT_ANALYSIS_PATH = PROJECT_ROOT / "outputs" / "transcript_analysis.json"
OUTPUT_PATH              = PROJECT_ROOT / "outputs" / "refined_caption.json"

assert PLOT_SUMMARIES_PATH.exists(),      f"plot_summaries.json not found: {PLOT_SUMMARIES_PATH}"
assert AGGREGATED_CAPTIONS_PATH.exists(), f"aggregated_caption.json not found: {AGGREGATED_CAPTIONS_PATH}"
assert TRANSCRIPT_ANALYSIS_PATH.exists(), f"transcript_analysis.json not found: {TRANSCRIPT_ANALYSIS_PATH}"

# ----------------------------
# Load inputs
# ----------------------------
print("Loading plot summaries...")
with open(PLOT_SUMMARIES_PATH, "r") as f:
    plot_summaries = json.load(f)

print("Loading aggregated captions...")
with open(AGGREGATED_CAPTIONS_PATH, "r") as f:
    all_aggregated_captions = json.load(f)

print("Loading transcript analyses...")
with open(TRANSCRIPT_ANALYSIS_PATH, "r") as f:
    all_transcript_analyses = json.load(f)

# ----------------------------
# Load existing refined captions (for resume)
# ----------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

if OUTPUT_PATH.exists():
    print(f"Found existing refined caption file at: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "r") as f:
        all_refined_captions = json.load(f)
    print("Resuming from existing progress.")
else:
    all_refined_captions = {}
    print("Starting fresh.")

# ----------------------------
# Helper: count pending clips
# Two upstream dependencies must both be present before a clip can be processed:
#   1. aggregated_caption  (from aggregated_caption.json)
#   2. transcript_analysis (from transcript_analysis.json)
# ----------------------------
def count_pending_clips(plot_summaries, all_aggregated_captions, all_transcript_analyses, all_refined_captions):
    total = 0
    pending = 0
    skipped_no_aggregated_caption = 0
    skipped_no_transcript_analysis = 0

    for movie_key, movie_data in plot_summaries.items():
        for clip in movie_data.get("clips", []):
            total += 1
            clip_id = clip["clip_id"]

            has_aggregated_caption = (
                movie_key in all_aggregated_captions and
                clip_id in all_aggregated_captions[movie_key] and
                "aggregated_caption" in all_aggregated_captions[movie_key][clip_id]
            )
            has_transcript_analysis = (
                movie_key in all_transcript_analyses and
                clip_id in all_transcript_analyses[movie_key] and
                "transcript_analysis" in all_transcript_analyses[movie_key][clip_id]
            )

            if not has_aggregated_caption:
                skipped_no_aggregated_caption += 1
                continue
            if not has_transcript_analysis:
                skipped_no_transcript_analysis += 1
                continue

            already_done = (
                movie_key in all_refined_captions and
                clip_id in all_refined_captions[movie_key] and
                "refined_caption" in all_refined_captions[movie_key][clip_id]
            )
            if not already_done:
                pending += 1

    return total, pending, skipped_no_aggregated_caption, skipped_no_transcript_analysis

(
    total_clips,
    pending_clips,
    skipped_no_aggregated_caption,
    skipped_no_transcript_analysis,
) = count_pending_clips(
    plot_summaries, all_aggregated_captions, all_transcript_analyses, all_refined_captions
)
already_done = (
    total_clips
    - pending_clips
    - skipped_no_aggregated_caption
    - skipped_no_transcript_analysis
)

print(f"\nTotal clips                       : {total_clips}")
print(f"Already refined                   : {already_done}")
print(f"Pending                           : {pending_clips}")
print(f"Skipped (no aggregated caption)   : {skipped_no_aggregated_caption}")
print(f"Skipped (no transcript analysis)  : {skipped_no_transcript_analysis}")

if pending_clips == 0:
    print("\nAll available clips already refined. Nothing to do.")
    raise SystemExit(0)

# ----------------------------
# Helper: atomic save
# ----------------------------
def save_progress(all_refined_captions, output_path: Path):
    tmp_path = output_path.with_suffix(".tmp.json")
    with open(tmp_path, "w") as f:
        json.dump(all_refined_captions, f, indent=2)
    tmp_path.replace(output_path)

# ----------------------------
# GPU info
# ----------------------------
print("\n===== GPU INFO =====")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())

# ----------------------------
# Load LLM ONCE
# ----------------------------
llm_model_name = (
    f"{PROJECT_DIR}/hf_cache/hub/"
    "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28"
)

print("\nLoading LLM (loaded once for all clips)...")
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    torch_dtype="auto",
    device_map="auto",
)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
print("LLM loaded.")

device_counts = defaultdict(int)
for _, param in llm_model.named_parameters():
    device_counts[str(param.device)] += 1
print("\n===== PARAM DEVICE DISTRIBUTION =====")
for dev, count in device_counts.items():
    print(f"  {dev}: {count} params")

# ----------------------------
# LLM inference helper
# ----------------------------
def run_llm_inference(system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = llm_tokenizer([text], return_tensors="pt").to(llm_model.device)

    generated_ids = llm_model.generate(**model_inputs, max_new_tokens=max_tokens)
    generated_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# ----------------------------
# Prompts
# ----------------------------
FINAL_ANALYSIS_SYSTEM = """You are a film analyst. Synthesize visual and dialogue information to describe a movie scene.
Be FACTUAL and CONCISE. DO NOT REFER TO ANY SPEAKER. Combine the provided information into a coherent description.
Write in 3-5 complete sentences."""

ACTION_PROMPT = """Based on the following information, describe what is PHYSICALLY HAPPENING in this scene.

VISUAL ACTION SEQUENCE:
{action_sequence}

SUBJECTS IN SCENE:
{subjects}

ACTIONS FROM DIALOGUE:
{key_actions_events}

CORRECTIONS/REVELATIONS FROM DIALOGUE:
{corrections_revelations}

Your task is to SYNTHESIZE the visual and dialogue information to describe:
- What actions are being performed?
- What events are unfolding?
- What is physically happening?

If the dialogue reveals something different from what the visuals suggest, incorporate that correction.

Write 3-5 sentences describing the action.

Rules:
- Combine visual and dialogue information coherently
- If dialogue corrects or clarifies visual information, use the correction
- Describe people by appearance, not by assumed names
- Be specific but avoid speculation
- Description must be solely based on the details provided. DO NOT RECREATE OR MISREPRESENT ANY SCENE OR ACTION."""

EMOTIONAL_TONE_PROMPT = """Based on the following information, describe the EMOTIONAL TONE of this scene.

VISUAL EMOTIONAL PROGRESSION:
{emotional_progression}

EMOTIONAL TONE FROM DIALOGUE:
{dialogue_emotional_tone}

Your task is to SYNTHESIZE the visual mood and dialogue tone to describe:
- What emotions does this scene convey?
- What should the audience feel?
- How do the visual mood and dialogue tone work together?

Write 3-5 sentences describing the emotional tone.

Rules:
- Combine visual and dialogue emotional information
- Consider both the atmosphere shown and the emotions expressed in speech
- Be specific but avoid speculation"""

THEMATIC_CONNECTION_PROMPT = """Based on the following information, describe how this scene connects to the movie's themes.

WHAT HAPPENS IN THE SCENE:
{action_sequence}

EMOTIONAL PROGRESSION:
{emotional_progression}

KEY EVENTS FROM DIALOGUE:
{key_actions_events}

CONTEXTUAL DETAILS:
{contextual_details}

MOVIE THEMES: {themes}

Your task is to analyze how this scene connects to the movie's themes:
- How does what happens in this scene relate to the themes?
- What might this scene represent or symbolize?
- How do the events and emotions connect to the broader story?

Write 3-5 sentences describing the thematic connection.

Rules:
- Base your analysis on the provided scene information
- Connect specifically to the listed themes
- Be thoughtful but avoid over-interpretation
- Ground your analysis in what actually happens in the scene"""

# ----------------------------
# Helper: refine one clip
# ----------------------------
def refine_clip(aggregated_caption: dict, transcript_analysis: dict, movie_themes: str) -> dict:
    """
    Run all three final analysis prompts for a single clip.
    Pulls fields from aggregated_caption and transcript_analysis dicts.
    """
    aspects = [
        (
            "action",
            ACTION_PROMPT.format(
                action_sequence      = aggregated_caption.get("action_sequence", "Not available"),
                subjects             = aggregated_caption.get("subjects", "Not available"),
                key_actions_events   = transcript_analysis.get("key_actions_events", "Not available"),
                corrections_revelations = transcript_analysis.get("corrections_revelations", "Not available"),
            ),
            300,
        ),
        (
            "emotional_tone",
            EMOTIONAL_TONE_PROMPT.format(
                emotional_progression  = aggregated_caption.get("emotional_progression", "Not available"),
                dialogue_emotional_tone = transcript_analysis.get("emotional_tone", "Not available"),
            ),
            300,
        ),
        (
            "thematic_connection",
            THEMATIC_CONNECTION_PROMPT.format(
                action_sequence    = aggregated_caption.get("action_sequence", "Not available"),
                emotional_progression = aggregated_caption.get("emotional_progression", "Not available"),
                key_actions_events = transcript_analysis.get("key_actions_events", "Not available"),
                contextual_details = transcript_analysis.get("contextual_details", "Not available"),
                themes             = movie_themes,
            ),
            300,
        ),
    ]

    final_scene_analysis = {}
    for aspect_name, user_prompt, max_tokens in aspects:
        print(f"    Generating {aspect_name}...")
        t = datetime.datetime.now()

        result = run_llm_inference(
            system_prompt=FINAL_ANALYSIS_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )

        elapsed = datetime.datetime.now() - t
        final_scene_analysis[aspect_name] = result.strip()
        print(f"      Done in {elapsed} | Preview: {result[:80].strip()}{'...' if len(result) > 80 else ''}")

    return final_scene_analysis

# ----------------------------
# Main loop
# ----------------------------
processed = 0
failed = []

for movie_key, movie_data in plot_summaries.items():
    clips = movie_data.get("clips", [])
    if not clips:
        continue

    # Movie-level themes — fall back gracefully if not present in plot_summaries
    movie_themes = movie_data.get("themes", "Not specified")

    if movie_key not in all_refined_captions:
        all_refined_captions[movie_key] = {}

    for clip in clips:
        clip_id       = clip["clip_id"]
        clip_filepath = clip["filepath"]

        # --- Dependency check: aggregated caption ---
        has_aggregated_caption = (
            movie_key in all_aggregated_captions and
            clip_id in all_aggregated_captions[movie_key] and
            "aggregated_caption" in all_aggregated_captions[movie_key][clip_id]
        )
        if not has_aggregated_caption:
            print(f"  [SKIP] {movie_key} / {clip_id} — no aggregated caption available yet.")
            continue

        # --- Dependency check: transcript analysis ---
        has_transcript_analysis = (
            movie_key in all_transcript_analyses and
            clip_id in all_transcript_analyses[movie_key] and
            "transcript_analysis" in all_transcript_analyses[movie_key][clip_id]
        )
        if not has_transcript_analysis:
            print(f"  [SKIP] {movie_key} / {clip_id} — no transcript analysis available yet.")
            continue

        # --- Resume check ---
        if (
            clip_id in all_refined_captions[movie_key] and
            "refined_caption" in all_refined_captions[movie_key][clip_id]
        ):
            print(f"  [SKIP] {movie_key} / {clip_id} — already refined.")
            continue

        print(f"\n[{processed + 1}/{pending_clips}] {movie_key} / {clip_id}")

        aggregated_caption  = all_aggregated_captions[movie_key][clip_id]["aggregated_caption"]
        transcript_analysis = all_transcript_analyses[movie_key][clip_id]["transcript_analysis"]

        t_start = datetime.datetime.now()
        try:
            final_scene_analysis = refine_clip(aggregated_caption, transcript_analysis, movie_themes)

            # Build output entry: preserve all clip metadata, add refined_caption
            all_refined_captions[movie_key][clip_id] = {
                **clip,                                   # clip_id, filepath, and any other metadata
                "refined_caption": final_scene_analysis,
            }

            elapsed = datetime.datetime.now() - t_start
            print(f"  Clip done in {elapsed}.")

            # Save after EVERY clip — progress is never lost
            save_progress(all_refined_captions, OUTPUT_PATH)
            print(f"  Progress saved → {OUTPUT_PATH}")

        except Exception as e:
            elapsed = datetime.datetime.now() - t_start
            print(f"  [ERROR] Failed after {elapsed}: {e}")
            failed.append({"movie": movie_key, "clip_id": clip_id, "reason": str(e)})
            save_progress(all_refined_captions, OUTPUT_PATH)

        processed += 1

# ----------------------------
# Cleanup
# ----------------------------
del llm_model
del llm_tokenizer
torch.cuda.empty_cache()
print("\nGPU memory cleared.")

# ----------------------------
# Summary
# ----------------------------
print("\n" + "=" * 60)
print("BATCH CAPTION REFINEMENT COMPLETE")
print("=" * 60)
print(f"  Processed : {processed}")
print(f"  Failed    : {len(failed)}")
if failed:
    print("\n  Failed clips:")
    for f in failed:
        print(f"    - {f['movie']} / {f['clip_id']}: {f['reason']}")
print(f"\n  Output saved to: {OUTPUT_PATH}")