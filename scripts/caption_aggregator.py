import os

from dotenv import load_dotenv

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
PLOT_SUMMARIES_PATH  = PROJECT_ROOT / "data" / "plot_summaries.json"
FRAME_CAPTIONS_PATH  = PROJECT_ROOT / "outputs" / "clip_frame_captions.json"
OUTPUT_PATH          = PROJECT_ROOT / "outputs" / "aggregated_caption.json"

assert PLOT_SUMMARIES_PATH.exists(),  f"plot_summaries.json not found: {PLOT_SUMMARIES_PATH}"
assert FRAME_CAPTIONS_PATH.exists(),  f"clip_frame_captions.json not found: {FRAME_CAPTIONS_PATH}"

# ----------------------------
# Load inputs
# ----------------------------
print("Loading plot summaries...")
with open(PLOT_SUMMARIES_PATH, "r") as f:
    plot_summaries = json.load(f)

print("Loading frame captions...")
with open(FRAME_CAPTIONS_PATH, "r") as f:
    all_frame_captions = json.load(f)

# ----------------------------
# Load existing aggregations (for resume)
# ----------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

if OUTPUT_PATH.exists():
    print(f"Found existing aggregation file at: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "r") as f:
        all_aggregations = json.load(f)
    print("Resuming from existing progress.")
else:
    all_aggregations = {}
    print("Starting fresh.")

# ----------------------------
# Helper: count pending clips
# ----------------------------
def count_pending_clips(plot_summaries, all_frame_captions, all_aggregations):
    total, pending, skipped_no_captions = 0, 0, 0
    for movie_key, movie_data in plot_summaries.items():
        for clip in movie_data.get("clips", []):
            total += 1
            clip_id = clip["clip_id"]

            # Can't aggregate if frame captions don't exist yet
            has_captions = (
                movie_key in all_frame_captions and
                clip_id in all_frame_captions[movie_key] and
                "frame_captions" in all_frame_captions[movie_key][clip_id]
            )
            if not has_captions:
                skipped_no_captions += 1
                continue

            already_done = (
                movie_key in all_aggregations and
                clip_id in all_aggregations[movie_key] and
                "aggregated_caption" in all_aggregations[movie_key][clip_id]
            )
            if not already_done:
                pending += 1

    return total, pending, skipped_no_captions

total_clips, pending_clips, skipped_no_captions = count_pending_clips(
    plot_summaries, all_frame_captions, all_aggregations
)
already_done = total_clips - pending_clips - skipped_no_captions

print(f"\nTotal clips           : {total_clips}")
print(f"Already aggregated    : {already_done}")
print(f"Pending               : {pending_clips}")
print(f"Skipped (no captions) : {skipped_no_captions}")

if pending_clips == 0:
    print("\nAll captioned clips already aggregated. Nothing to do.")
    raise SystemExit(0)

# ----------------------------
# Helper: atomic save
# ----------------------------
def save_progress(all_aggregations, output_path: Path):
    tmp_path = output_path.with_suffix(".tmp.json")
    with open(tmp_path, "w") as f:
        json.dump(all_aggregations, f, indent=2)
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
AGGREGATION_SYSTEM_PROMPT = """You are an expert at analyzing video scenes. Synthesize frame-by-frame descriptions into a coherent description.
Describe ONLY what is observable. Do not invent details. Do not reference specific movies or guess character names.
Write in 4-6 complete sentences covering information from all scene frames."""

VISUAL_DESCRIPTION_PROMPT = """Based on these frame descriptions, write a VISUAL DESCRIPTION of the scene.

FRAME DESCRIPTIONS (chronological order):
{frame_descriptions}

Describe in 4-6 sentences covering:
- Setting/location (describe only what you see)
- Lighting and color palette
- Cinematography style (close-ups, wide shots, etc.)
- Notable visual details (objects, text, environmental elements)

Rules:
- Only include information present in the frame descriptions
- Do NOT reference specific movies or franchises by name
- Be specific but avoid speculation
- Write as flowing prose, not bullet points"""

SUBJECTS_PROMPT = """Based on these frame descriptions, identify and describe all SUBJECTS (people and significant objects) in the scene.

FRAME DESCRIPTIONS (chronological order):
{frame_descriptions}

Describe in 4-6 complete sentences covering:

For people:
- Describe each unique person ONCE based on their appearance
- Use visual features to identify if the same person appears in multiple frames (same clothing, hair, build = likely same person)
- Describe: clothing, hair, visible features, accessories
- Do NOT repeat the same person multiple times
- Do NOT guess character names

For objects:
- Include significant objects that play a role in the scene

Rules:
- Only include information present in the frame descriptions
- Same person in multiple frames = ONE description with combined details
- Describe people by appearance, not by assumed identity
- Be specific but avoid speculation
- Write as flowing prose, not bullet points"""

ACTION_SEQUENCE_PROMPT = """Based on these frame descriptions, write the ACTION SEQUENCE describing what happens in the scene.

FRAME DESCRIPTIONS (chronological order):
{frame_descriptions}

Your task is to READ and UNDERSTAND each frame description, then WEAVE them together into a COHERENT NARRATIVE that makes sense.

Guidelines:
- Read all frame descriptions carefully
- Identify what is happening across the frames
- Find connections and continuity between frames
- Stitch the descriptions together into a logical, flowing narrative
- Write in prose form, like a screenplay description (4-6 sentences)
- When referring to people, describe them briefly by appearance
- Use transitions: "The scene opens with...", "Then...", "Meanwhile...", "As this happens...", "The scene concludes with..."

Key principles:
- Make sense of ALL the frame descriptions provided
- Create a narrative that flows logically
- If frames show different moments or perspectives, connect them sensibly
- Ensure your narrative accounts for what is shown in the FIRST frames through to the LAST frames

Do NOT:
- Write a frame-by-frame list
- Ignore or skip any frame descriptions
- Guess character names
- Invent details not present in the descriptions

Rules:
- Only include information present in the frame descriptions
- Focus on creating a coherent story from the visual snapshots
- Be specific but avoid speculation"""

EMOTIONAL_PROGRESSION_PROMPT = """Based on these frame descriptions, describe the EMOTIONAL PROGRESSION of the scene.

FRAME DESCRIPTIONS (chronological order):
{frame_descriptions}

Describe in 4-6 sentences the overall emotional arc of the SCENE as a whole.

Consider:
- What is the starting mood/atmosphere?
- How does it shift or evolve through the scene?
- What is the ending mood/atmosphere?

If specific individuals show notable emotional states (e.g., someone crying, someone visibly angry), describe WHAT is happening without trying to explain WHO they are. Refer to them by appearance.

Example approach: "The scene begins with a tense atmosphere. As events unfold, one person with curly hair appears distressed. The overall mood shifts from tension to sorrow."

Rules:
- Only include information present in the frame descriptions
- Focus on the scene's mood, not character-by-character analysis
- Describe people by appearance if mentioning their emotions
- Do NOT guess character names
- Be specific but avoid speculation
- Write as flowing prose, not bullet points"""

ASPECTS = [
    ("visual_description",   VISUAL_DESCRIPTION_PROMPT,   500),
    ("subjects",             SUBJECTS_PROMPT,             500),
    ("action_sequence",      ACTION_SEQUENCE_PROMPT,      600),
    ("emotional_progression",EMOTIONAL_PROGRESSION_PROMPT,500),
]

# ----------------------------
# Helper: aggregate one clip
# ----------------------------
def aggregate_clip(frame_captions: dict) -> dict:
    """
    Run all four LLM aspect prompts for a single clip.
    frame_captions: the frame_dict from clip_frame_captions.json
                    (values are either str or {"description":..., "path":...})
    Returns aggregated_scene_description dict.
    """
    # Normalize: support both plain-string captions and {"description", "path"} dicts
    descriptions_only = {
        k: (v["description"] if isinstance(v, dict) else v)
        for k, v in frame_captions.items()
    }
    frame_descriptions_formatted = json.dumps(descriptions_only, indent=2)

    aggregated = {}
    for aspect_name, prompt_template, max_tokens in ASPECTS:
        print(f"    Generating {aspect_name}...")
        t = datetime.datetime.now()

        result = run_llm_inference(
            system_prompt=AGGREGATION_SYSTEM_PROMPT,
            user_prompt=prompt_template.format(
                frame_descriptions=frame_descriptions_formatted
            ),
            max_tokens=max_tokens,
        )

        elapsed = datetime.datetime.now() - t
        aggregated[aspect_name] = result.strip()
        print(f"      Done in {elapsed} | Preview: {result[:80].strip()}{'...' if len(result) > 80 else ''}")

    return aggregated

# ----------------------------
# Main loop
# ----------------------------
processed = 0
failed = []

for movie_key, movie_data in plot_summaries.items():
    clips = movie_data.get("clips", [])
    if not clips:
        continue

    if movie_key not in all_aggregations:
        all_aggregations[movie_key] = {}

    for clip in clips:
        clip_id       = clip["clip_id"]
        clip_filepath = clip["filepath"]

        # --- Skip if frame captions are missing ---
        has_captions = (
            movie_key in all_frame_captions and
            clip_id in all_frame_captions[movie_key] and
            "frame_captions" in all_frame_captions[movie_key][clip_id]
        )
        if not has_captions:
            print(f"  [SKIP] {movie_key} / {clip_id} — no frame captions available yet.")
            continue

        # --- Resume check ---
        if (
            clip_id in all_aggregations[movie_key] and
            "aggregated_caption" in all_aggregations[movie_key][clip_id]
        ):
            print(f"  [SKIP] {movie_key} / {clip_id} — already aggregated.")
            continue

        print(f"\n[{processed + 1}/{pending_clips}] {movie_key} / {clip_id}")

        t_start = datetime.datetime.now()
        try:
            frame_captions = all_frame_captions[movie_key][clip_id]["frame_captions"]
            aggregated_scene_description = aggregate_clip(frame_captions)

            # Build output entry: preserve all clip metadata, add aggregated_caption
            all_aggregations[movie_key][clip_id] = {
                **clip,                                             # clip_id, filepath, and any other metadata
                "aggregated_caption": aggregated_scene_description,
            }

            elapsed = datetime.datetime.now() - t_start
            print(f"  Clip done in {elapsed}.")

            # Save after EVERY clip — progress is never lost
            save_progress(all_aggregations, OUTPUT_PATH)
            print(f"  Progress saved → {OUTPUT_PATH}")

        except Exception as e:
            elapsed = datetime.datetime.now() - t_start
            print(f"  [ERROR] Failed after {elapsed}: {e}")
            failed.append({"movie": movie_key, "clip_id": clip_id, "reason": str(e)})
            save_progress(all_aggregations, OUTPUT_PATH)

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
print("BATCH CAPTION AGGREGATION COMPLETE")
print("=" * 60)
print(f"  Processed : {processed}")
print(f"  Failed    : {len(failed)}")
if failed:
    print("\n  Failed clips:")
    for f in failed:
        print(f"    - {f['movie']} / {f['clip_id']}: {f['reason']}")
print(f"\n  Output saved to: {OUTPUT_PATH}")