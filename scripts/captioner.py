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

import sys
import json
import datetime
import torch
from pathlib import Path
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ----------------------------
# Project imports
# ----------------------------
PROJECT_ROOT = Path(PROJECT_DIR)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from indexing.frame_extractor import FrameExtractor

# ----------------------------
# Paths
# ----------------------------
PLOT_SUMMARIES_PATH = PROJECT_ROOT / "data" / "plot_summaries.json"
OUTPUT_PATH         = PROJECT_ROOT / "outputs" / "clip_frame_captions.json"
FRAMES_ROOT         = PROJECT_ROOT / "outputs" / "frames"   # outputs/frames/<movie>/<clip_id>/

assert PLOT_SUMMARIES_PATH.exists(), f"plot_summaries.json not found: {PLOT_SUMMARIES_PATH}"

# ----------------------------
# GPU info
# ----------------------------
print("\n===== GPU INFO =====")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())

# ----------------------------
# Load plot summaries
# ----------------------------
print("\nLoading plot summaries...")
with open(PLOT_SUMMARIES_PATH, "r") as f:
    plot_summaries = json.load(f)

# ----------------------------
# Load existing captions (for resume)
# ----------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

if OUTPUT_PATH.exists():
    print(f"Found existing captions file at: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "r") as f:
        all_captions = json.load(f)
    print("Resuming from existing progress.")
else:
    all_captions = {}
    print("Starting fresh.")

# ----------------------------
# Helper: count pending clips
# ----------------------------
def count_pending_clips(plot_summaries, all_captions):
    total, pending = 0, 0
    for movie_key, movie_data in plot_summaries.items():
        for clip in movie_data.get("clips", []):
            total += 1
            clip_id = clip["clip_id"]
            already_done = (
                movie_key in all_captions and
                clip_id in all_captions[movie_key] and
                "frame_captions" in all_captions[movie_key][clip_id]
            )
            if not already_done:
                pending += 1
    return total, pending

total_clips, pending_clips = count_pending_clips(plot_summaries, all_captions)
print(f"\nTotal clips : {total_clips}")
print(f"Already done: {total_clips - pending_clips}")
print(f"Pending     : {pending_clips}")

if pending_clips == 0:
    print("\nAll clips already captioned. Nothing to do.")
    sys.exit(0)

# ----------------------------
# Helper: atomic save
# ----------------------------
def save_progress(all_captions, output_path: Path):
    tmp_path = output_path.with_suffix(".tmp.json")
    with open(tmp_path, "w") as f:
        json.dump(all_captions, f, indent=2)
    tmp_path.replace(output_path)   # atomic rename — never leaves a half-written file

# ----------------------------
# Helper: caption all frames for one clip
# ----------------------------
def caption_clip(clip_path: Path, movie_key: str, clip_id: str,
                 vlm_model, vlm_processor, frame_extractor) -> dict:
    """
    Extract frames, save them to disk, run VLM captioning, and return frame_dict.
    frame_dict structure:
        {
            "frame_0": {"description": "...", "path": "outputs/frames/<movie>/<clip>/frame_0.png"},
            ...
        }
    """
    # Where to store PNG frames for this clip
    frame_dir = FRAMES_ROOT / movie_key / clip_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames
    frames, video_meta = frame_extractor.extract_frames(str(clip_path))
    print(f"  Extracted {len(frames)} frames.")

    frame_dict = {}

    for frame_idx, frame in enumerate(frames):
        frame = frame.convert("RGB")
        frame_key = f"frame_{frame_idx}"

        # Relative path stored in JSON (human-readable, portable)
        rel_frame_path = str(
            (FRAMES_ROOT / movie_key / clip_id / f"{frame_key}.png")
            .relative_to(PROJECT_ROOT)
        )
        abs_frame_path = PROJECT_ROOT / rel_frame_path

        # Save PNG to disk
        frame.save(abs_frame_path)

        print(f"  [{frame_idx + 1}/{len(frames)}] Captioning {frame_key}...")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": frame,   # PIL image passed directly
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in 3-5 complete sentences. "
                            "If people are visible, describe their facial expressions as specifically as possible. "
                            "DO NOT SPECULATE. ONLY DESCRIBE WHAT YOU SEE."
                        ),
                    },
                ],
            }
        ]

        text = vlm_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        t_start = datetime.datetime.now()

        generated_ids = vlm_model.generate(**inputs, max_new_tokens=256)

        elapsed = datetime.datetime.now() - t_start

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = vlm_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        caption = output_text[0]
        print(f"    Inference: {elapsed} | Caption: {caption[:80]}{'...' if len(caption) > 80 else ''}")

        frame_dict[frame_key] = {
            "description": caption,
            "path": rel_frame_path,
        }

    return frame_dict

# ----------------------------
# Load model ONCE
# ----------------------------
model_name = (
    f"{PROJECT_DIR}/hf_cache/hub/"
    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
    "cc594898137f460bfe9f0759e9844b3ce807cfb5"
)

print("\nLoading VLM model (loaded once for all clips)...")
vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
print("Model loaded.")

vlm_processor = AutoProcessor.from_pretrained(model_name)
print("Processor loaded.")

# ----------------------------
# Model placement diagnostics
# ----------------------------
print("\n===== MODEL DEVICE MAP =====")
print(vlm_model.hf_device_map if hasattr(vlm_model, "hf_device_map") else "No hf_device_map attribute")

device_counts = defaultdict(int)
for _, param in vlm_model.named_parameters():
    device_counts[str(param.device)] += 1
print("\n===== PARAM DEVICE DISTRIBUTION =====")
for dev, count in device_counts.items():
    print(f"  {dev}: {count} params")

# ----------------------------
# Frame extractor (shared, stateless)
# ----------------------------
frame_extractor = FrameExtractor(
    num_frames=12,
    sampling_strategy="keyframe",
)

# ----------------------------
# Main loop
# ----------------------------
processed = 0
failed = []

for movie_key, movie_data in plot_summaries.items():
    clips = movie_data.get("clips", [])
    if not clips:
        continue

    if movie_key not in all_captions:
        all_captions[movie_key] = {}

    for clip in clips:
        clip_id       = clip["clip_id"]
        clip_filepath = clip["filepath"]   # e.g. "clips/2012/massive_earthquake.mp4"

        # --- Resume check ---
        if (
            clip_id in all_captions[movie_key] and
            "frame_captions" in all_captions[movie_key][clip_id]
        ):
            print(f"  [SKIP] {movie_key} / {clip_id} — already captioned.")
            continue

        clip_path = PROJECT_ROOT / clip_filepath

        if not clip_path.exists():
            print(f"  [WARN] Clip not found, skipping: {clip_path}")
            failed.append({"movie": movie_key, "clip_id": clip_id, "reason": "file not found"})
            continue

        print(f"\n[{processed + 1}/{pending_clips}] {movie_key} / {clip_id}")
        print(f"  Path: {clip_path}")

        t_start = datetime.datetime.now()
        try:
            frame_dict = caption_clip(
                clip_path, movie_key, clip_id,
                vlm_model, vlm_processor, frame_extractor
            )

            # Build output entry: preserve all clip metadata, add frame_captions
            all_captions[movie_key][clip_id] = {
                **clip,                      # clip_id, filepath, and any other metadata fields
                "frame_captions": frame_dict,
            }

            elapsed = datetime.datetime.now() - t_start
            print(f"  Clip done in {elapsed}. Frames captioned: {len(frame_dict)}")

            # Save after EVERY clip — progress is never lost
            save_progress(all_captions, OUTPUT_PATH)
            print(f"  Progress saved → {OUTPUT_PATH}")

        except Exception as e:
            elapsed = datetime.datetime.now() - t_start
            print(f"  [ERROR] Failed after {elapsed}: {e}")
            failed.append({"movie": movie_key, "clip_id": clip_id, "reason": str(e)})
            save_progress(all_captions, OUTPUT_PATH)  # save any other completed clips

        processed += 1

# ----------------------------
# Cleanup
# ----------------------------
del vlm_model
del vlm_processor
torch.cuda.empty_cache()
print("\nGPU memory cleared.")

# ----------------------------
# Summary
# ----------------------------
print("\n" + "=" * 60)
print("BATCH FRAME CAPTIONING COMPLETE")
print("=" * 60)
print(f"  Processed : {processed}")
print(f"  Failed    : {len(failed)}")
if failed:
    print("\n  Failed clips:")
    for f in failed:
        print(f"    - {f['movie']} / {f['clip_id']}: {f['reason']}")
print(f"\n  Captions saved to : {OUTPUT_PATH}")
print(f"  Frames saved to   : {FRAMES_ROOT}")