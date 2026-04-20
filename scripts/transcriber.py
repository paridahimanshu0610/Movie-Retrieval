import os
from dotenv import load_dotenv

# ----------------------------
# Project directory (Grace)
# ----------------------------
PROJECT_DIR = "/Users/himanshu/Documents/TAMU/Courses/ISR/Project/Movie Retrieval"

# Load variables from .env into environment
load_dotenv()

import whisper
import torch
import soundfile as sf
import subprocess
import json
import datetime
from pyannote.audio import Pipeline
from pathlib import Path

# ----------------------------
# Setup
# ----------------------------
PROJECT_ROOT = Path(PROJECT_DIR)
PLOT_SUMMARIES_PATH = PROJECT_ROOT / "data" / "plot_summaries.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "clip_transcript.json"
WHISPER_CACHE = f"{PROJECT_DIR}/hf_cache/whisper"

assert PLOT_SUMMARIES_PATH.exists(), f"plot_summaries.json not found: {PLOT_SUMMARIES_PATH}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ----------------------------
# Load plot summaries
# ----------------------------
print("\nLoading plot summaries...")
with open(PLOT_SUMMARIES_PATH, "r") as f:
    plot_summaries = json.load(f)

# ----------------------------
# Load existing transcripts (for resume)
# ----------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

if OUTPUT_PATH.exists():
    print(f"Found existing transcript file at: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "r") as f:
        all_transcripts = json.load(f)
    print("Resuming from existing progress.")
else:
    all_transcripts = {}
    print("Starting fresh.")

# ----------------------------
# Helper: count pending clips
# ----------------------------
def count_pending_clips(plot_summaries, all_transcripts):
    total, pending = 0, 0
    for movie_key, movie_data in plot_summaries.items():
        for clip in movie_data.get("clips", []):
            total += 1
            clip_id = clip["clip_id"]
            already_done = (
                movie_key in all_transcripts and
                clip_id in all_transcripts[movie_key] and
                "transcript" in all_transcripts[movie_key][clip_id]
            )
            if not already_done:
                pending += 1
    return total, pending

total_clips, pending_clips = count_pending_clips(plot_summaries, all_transcripts)
print(f"\nTotal clips: {total_clips} | Already transcribed: {total_clips - pending_clips} | Pending: {pending_clips}")

if pending_clips == 0:
    print("All clips already transcribed. Nothing to do.")
    exit(0)

# ----------------------------
# Helper functions
# ----------------------------
def extract_audio(clip_path: Path, wav_path: Path):
    result = subprocess.run([
        "ffmpeg", "-y", "-i", str(clip_path),
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-af", "loudnorm,highpass=f=200,lowpass=f=3000",
        str(wav_path)
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error for {clip_path}:\n{result.stderr}")


def get_speaker_at_time(timestamp, diarization):
    if hasattr(diarization, 'speaker_diarization'):
        annotation = diarization.speaker_diarization
    elif hasattr(diarization, 'annotation'):
        annotation = diarization.annotation
    else:
        annotation = diarization

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        if turn.start <= timestamp <= turn.end:
            return speaker
    return "Unknown"


def merge_segments_by_speaker(segments, diarization):
    merged = []
    current_speaker = None
    current_text = []
    current_start = 0

    for segment in segments:
        mid_time = (segment["start"] + segment["end"]) / 2
        speaker = get_speaker_at_time(mid_time, diarization)
        text = segment["text"].strip()

        if speaker == current_speaker:
            current_text.append(text)
        else:
            if current_speaker is not None and current_text:
                merged.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_text),
                    "start": current_start
                })
            current_speaker = speaker
            current_text = [text]
            current_start = segment["start"]

    if current_text:
        merged.append({
            "speaker": current_speaker,
            "text": " ".join(current_text),
            "start": current_start
        })

    return merged


def format_transcript_for_prompt(transcript: list[dict]) -> str:
    lines = []
    for entry in transcript:
        speaker = entry.get("speaker", "Unknown")
        text = entry.get("text", "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def save_progress(all_transcripts, output_path):
    """Atomically save progress to disk."""
    tmp_path = output_path.with_suffix(".tmp.json")
    with open(tmp_path, "w") as f:
        json.dump(all_transcripts, f, indent=2)
    tmp_path.replace(output_path)  # atomic rename


def transcribe_clip(clip_path: Path, wav_path: Path, whisper_model, diarization_pipeline):
    """Full pipeline for a single clip. Returns (merged_segments, formatted_transcript)."""
    # Step 1: Extract audio
    extract_audio(clip_path, wav_path)

    # Step 2: Transcribe
    transcription = whisper_model.transcribe(
        str(wav_path),
        language="en",
        word_timestamps=True,
        fp16=device.type == "cuda"
    )

    # Step 3: Diarize
    audio_data, sample_rate = sf.read(str(wav_path))
    waveform = torch.tensor(audio_data).unsqueeze(0).float()
    diarization = diarization_pipeline({
        "waveform": waveform,
        "sample_rate": sample_rate
    })

    # Step 4: Merge & format
    merged_segments = merge_segments_by_speaker(transcription["segments"], diarization)
    formatted_transcript = format_transcript_for_prompt(merged_segments)

    # Cleanup temp wav
    wav_path.unlink(missing_ok=True)

    return merged_segments, formatted_transcript

# ----------------------------
# Load models ONCE
# ----------------------------
print("\nLoading Whisper model (loaded once for all clips)...")
whisper_model = whisper.load_model("large", device=device, download_root=WHISPER_CACHE)

print("Loading diarization pipeline (loaded once for all clips)...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=os.environ["HF_TOKEN"]
)
diarization_pipeline = diarization_pipeline.to(device)

# ----------------------------
# Main loop
# ----------------------------
processed = 0
failed = []

for movie_key, movie_data in plot_summaries.items():
    clips = movie_data.get("clips", [])
    if not clips:
        continue

    # Ensure movie entry exists in output
    if movie_key not in all_transcripts:
        all_transcripts[movie_key] = {}

    for clip in clips:
        clip_id = clip["clip_id"]
        clip_filepath = clip["filepath"]  # relative path, e.g. "clips/2012/massive_earthquake.mp4"

        # --- Resume check ---
        if (
            clip_id in all_transcripts[movie_key] and
            "transcript" in all_transcripts[movie_key][clip_id]
        ):
            print(f"  [SKIP] {movie_key} / {clip_id} — already transcribed.")
            continue

        clip_path = PROJECT_ROOT / clip_filepath
        wav_path = clip_path.with_suffix(".wav")

        if not clip_path.exists():
            print(f"  [WARN] Clip not found, skipping: {clip_path}")
            failed.append({"movie": movie_key, "clip_id": clip_id, "reason": "file not found"})
            continue

        print(f"\n[{processed + 1}/{pending_clips}] Processing: {movie_key} / {clip_id}")
        print(f"  Path: {clip_path}")

        t_start = datetime.datetime.now()
        try:
            merged_segments, formatted_transcript = transcribe_clip(
                clip_path, wav_path, whisper_model, diarization_pipeline
            )

            # Build output entry: copy all clip metadata, then add transcript fields
            all_transcripts[movie_key][clip_id] = {
                **clip,  # preserves clip_id, filepath, and any other metadata
                "transcript": merged_segments,
                "formatted_transcript": formatted_transcript,
            }

            elapsed = datetime.datetime.now() - t_start
            print(f"  Done in {elapsed}. Segments: {len(merged_segments)}")

            # Save after EVERY clip so progress is never lost
            save_progress(all_transcripts, OUTPUT_PATH)
            print(f"  Progress saved to: {OUTPUT_PATH}")

        except Exception as e:
            elapsed = datetime.datetime.now() - t_start
            print(f"  [ERROR] Failed after {elapsed}: {e}")
            failed.append({"movie": movie_key, "clip_id": clip_id, "reason": str(e)})
            # Still save so other completed clips aren't lost
            save_progress(all_transcripts, OUTPUT_PATH)

        processed += 1

# ----------------------------
# Cleanup
# ----------------------------
del whisper_model
del diarization_pipeline
torch.cuda.empty_cache()
print("\nGPU memory cleared.")

# ----------------------------
# Summary
# ----------------------------
print("\n" + "=" * 60)
print("BATCH TRANSCRIPTION COMPLETE")
print("=" * 60)
print(f"  Processed : {processed}")
print(f"  Failed    : {len(failed)}")
if failed:
    print("\n  Failed clips:")
    for f in failed:
        print(f"    - {f['movie']} / {f['clip_id']}: {f['reason']}")
print(f"\n  Output saved to: {OUTPUT_PATH}")