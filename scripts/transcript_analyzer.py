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
PLOT_SUMMARIES_PATH = PROJECT_ROOT / "data"    / "plot_summaries.json"
TRANSCRIPTS_PATH    = PROJECT_ROOT / "outputs" / "clip_transcript.json"
OUTPUT_PATH         = PROJECT_ROOT / "outputs" / "transcript_analysis.json"

assert PLOT_SUMMARIES_PATH.exists(), f"plot_summaries.json not found: {PLOT_SUMMARIES_PATH}"
assert TRANSCRIPTS_PATH.exists(),    f"clip_transcript.json not found: {TRANSCRIPTS_PATH}"

# ----------------------------
# Load inputs
# ----------------------------
print("Loading plot summaries...")
with open(PLOT_SUMMARIES_PATH, "r") as f:
    plot_summaries = json.load(f)

print("Loading transcripts...")
with open(TRANSCRIPTS_PATH, "r") as f:
    all_transcripts = json.load(f)

# ----------------------------
# Load existing analyses (for resume)
# ----------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

if OUTPUT_PATH.exists():
    print(f"Found existing analysis file at: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "r") as f:
        all_analyses = json.load(f)
    print("Resuming from existing progress.")
else:
    all_analyses = {}
    print("Starting fresh.")

# ----------------------------
# Helper: count pending clips
# ----------------------------
def count_pending_clips(plot_summaries, all_transcripts, all_analyses):
    total, pending, skipped_no_transcript = 0, 0, 0
    for movie_key, movie_data in plot_summaries.items():
        for clip in movie_data.get("clips", []):
            total += 1
            clip_id = clip["clip_id"]

            # Can't analyse if transcript doesn't exist yet
            has_transcript = (
                movie_key in all_transcripts and
                clip_id in all_transcripts[movie_key] and
                "formatted_transcript" in all_transcripts[movie_key][clip_id]
            )
            if not has_transcript:
                skipped_no_transcript += 1
                continue

            already_done = (
                movie_key in all_analyses and
                clip_id in all_analyses[movie_key] and
                "transcript_analysis" in all_analyses[movie_key][clip_id]
            )
            if not already_done:
                pending += 1

    return total, pending, skipped_no_transcript

total_clips, pending_clips, skipped_no_transcript = count_pending_clips(
    plot_summaries, all_transcripts, all_analyses
)
already_done = total_clips - pending_clips - skipped_no_transcript

print(f"\nTotal clips              : {total_clips}")
print(f"Already analysed         : {already_done}")
print(f"Pending                  : {pending_clips}")
print(f"Skipped (no transcript)  : {skipped_no_transcript}")

if pending_clips == 0:
    print("\nAll transcribed clips already analysed. Nothing to do.")
    raise SystemExit(0)

# ----------------------------
# Helper: atomic save
# ----------------------------
def save_progress(all_analyses, output_path: Path):
    tmp_path = output_path.with_suffix(".tmp.json")
    with open(tmp_path, "w") as f:
        json.dump(all_analyses, f, indent=2)
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
TRANSCRIPT_ANALYSIS_SYSTEM = """You are a dialogue analyst. Extract useful information from movie scene transcripts.
Be FACTUAL and CONCISE. Only extract what is clearly present in the dialogue.
Write in 2-4 complete sentences."""

EMOTIONAL_TONE_PROMPT = """Analyze this transcript and describe the EMOTIONAL TONE of the scene.

TRANSCRIPT:
{transcript}

Your task is to READ and UNDERSTAND the dialogue, then describe the emotional tone.

Consider:
- What is the overall mood conveyed through the dialogue? (calm, tense, urgent, humorous, sad, fearful, excited, etc.)
- Does the tone shift during the scene? If so, how?
- What emotions are the speakers expressing through their words?

Write 2-4 sentences describing the emotional tone.

Rules:
- Only include information clearly conveyed in the dialogue
- Focus on the mood and emotions present in the speech
- If emotional tone is not evident, write "Not evident from transcript."
- Do NOT guess or speculate beyond what is said"""

KEY_ACTIONS_PROMPT = """Analyze this transcript and identify KEY ACTIONS or EVENTS being discussed or performed.

TRANSCRIPT:
{transcript}

Your task is to READ and UNDERSTAND the dialogue, then identify what is happening.

Consider:
- What actions are characters performing or discussing?
- What events are unfolding or being referenced?
- What are people doing, planning, or reacting to?

Write 2-4 sentences describing the key actions and events.

Rules:
- Only include information clearly conveyed in the dialogue
- Focus on actions and events, not emotions or tone
- If no clear actions/events are evident, write "Not evident from transcript."
- Do NOT guess or speculate beyond what is said"""

CORRECTIONS_PROMPT = """Analyze this transcript and identify any CORRECTIONS or REVELATIONS.

TRANSCRIPT:
{transcript}

Your task is to READ and UNDERSTAND the dialogue, then identify moments where:
- Something is revealed to be different than expected
- A misconception is corrected (e.g., "that's not X, it's Y")
- New information changes understanding of the situation
- Someone realizes or discovers something important

Write 2-4 sentences describing any corrections or revelations.

Rules:
- Only include information clearly conveyed in the dialogue
- Focus on moments of realization, correction, or revelation
- If no corrections/revelations are evident, write "Not evident from transcript."
- Do NOT guess or speculate beyond what is said"""

CONTEXTUAL_DETAILS_PROMPT = """Analyze this transcript and extract CONTEXTUAL DETAILS that help understand the scene.

TRANSCRIPT:
{transcript}

Your task is to READ and UNDERSTAND the dialogue, then identify contextual information.

Consider:
- Any mentioned objects, locations, or places
- Any threats, dangers, or stakes mentioned
- Any time pressure or urgency expressed
- Any references to people, things, or events
- Any background information provided through dialogue

Write 2-4 sentences describing relevant contextual details.

Rules:
- Only include information clearly conveyed in the dialogue
- Focus on factual details that provide context
- If no contextual details are evident, write "Not evident from transcript."
- Do NOT guess or speculate beyond what is said"""

TRANSCRIPT_ASPECTS = [
    ("emotional_tone",          EMOTIONAL_TONE_PROMPT,      200),
    ("key_actions_events",      KEY_ACTIONS_PROMPT,         200),
    ("corrections_revelations", CORRECTIONS_PROMPT,         200),
    ("contextual_details",      CONTEXTUAL_DETAILS_PROMPT,  200),
]

# ----------------------------
# Helper: analyse one clip's transcript
# ----------------------------
def analyse_transcript(formatted_transcript: str) -> dict:
    """
    Run all four LLM aspect prompts for a single clip's transcript.
    Handles empty/too-short transcripts gracefully, matching the
    fallback behaviour of the original single-clip script.
    """
    NOT_AVAILABLE = "Not available - no transcript"

    if not formatted_transcript or len(formatted_transcript.strip()) < 10:
        print("    WARNING: Transcript is empty or too short. Returning fallback values.")
        return {
            "emotional_tone":          NOT_AVAILABLE,
            "key_actions_events":      NOT_AVAILABLE,
            "corrections_revelations": NOT_AVAILABLE,
            "contextual_details":      NOT_AVAILABLE,
        }

    analysis = {}
    for aspect_name, prompt_template, max_tokens in TRANSCRIPT_ASPECTS:
        print(f"    Extracting {aspect_name}...")
        t = datetime.datetime.now()

        result = run_llm_inference(
            system_prompt=TRANSCRIPT_ANALYSIS_SYSTEM,
            user_prompt=prompt_template.format(transcript=formatted_transcript),
            max_tokens=max_tokens,
        )

        elapsed = datetime.datetime.now() - t
        analysis[aspect_name] = result.strip()
        print(f"      Done in {elapsed} | Preview: {result[:80].strip()}{'...' if len(result) > 80 else ''}")

    return analysis

# ----------------------------
# Main loop
# ----------------------------
processed = 0
failed = []

for movie_key, movie_data in plot_summaries.items():
    clips = movie_data.get("clips", [])
    if not clips:
        continue

    if movie_key not in all_analyses:
        all_analyses[movie_key] = {}

    for clip in clips:
        clip_id       = clip["clip_id"]
        clip_filepath = clip["filepath"]

        # --- Skip if transcript is missing ---
        has_transcript = (
            movie_key in all_transcripts and
            clip_id in all_transcripts[movie_key] and
            "formatted_transcript" in all_transcripts[movie_key][clip_id]
        )
        if not has_transcript:
            print(f"  [SKIP] {movie_key} / {clip_id} — no transcript available yet.")
            continue

        # --- Resume check ---
        if (
            clip_id in all_analyses[movie_key] and
            "transcript_analysis" in all_analyses[movie_key][clip_id]
        ):
            print(f"  [SKIP] {movie_key} / {clip_id} — already analysed.")
            continue

        print(f"\n[{processed + 1}/{pending_clips}] {movie_key} / {clip_id}")

        # Use the pre-formatted transcript stored by the transcription script directly
        formatted_transcript = all_transcripts[movie_key][clip_id]["formatted_transcript"]
        print(f"  Transcript length: {len(formatted_transcript)} characters")

        t_start = datetime.datetime.now()
        try:
            transcript_analysis = analyse_transcript(formatted_transcript)

            # Build output entry: preserve all clip metadata, add transcript_analysis
            all_analyses[movie_key][clip_id] = {
                **clip,                                 # clip_id, filepath, and any other metadata
                "transcript_analysis": transcript_analysis,
            }

            elapsed = datetime.datetime.now() - t_start
            print(f"  Clip done in {elapsed}.")

            # Save after EVERY clip — progress is never lost
            save_progress(all_analyses, OUTPUT_PATH)
            print(f"  Progress saved → {OUTPUT_PATH}")

        except Exception as e:
            elapsed = datetime.datetime.now() - t_start
            print(f"  [ERROR] Failed after {elapsed}: {e}")
            failed.append({"movie": movie_key, "clip_id": clip_id, "reason": str(e)})
            save_progress(all_analyses, OUTPUT_PATH)

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
print("BATCH TRANSCRIPT ANALYSIS COMPLETE")
print("=" * 60)
print(f"  Processed : {processed}")
print(f"  Failed    : {len(failed)}")
if failed:
    print("\n  Failed clips:")
    for f in failed:
        print(f"    - {f['movie']} / {f['clip_id']}: {f['reason']}")
print(f"\n  Output saved to: {OUTPUT_PATH}")