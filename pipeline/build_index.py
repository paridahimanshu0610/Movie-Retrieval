"""
build_index.py — Ingest clips from CLIPS_DIR, generate embeddings, save index.

Usage
─────
    python pipeline/build_index.py

Expects .mp4 files inside ./clips/.
Naming convention: <movie_title>_<clip_id>.mp4
  e.g.  interstellar_01.mp4   interstellar_02.mp4   gravity_01.mp4

After running, the index is persisted in ./index/ as:
  visual.npy | text.npy | metadata.json
"""

import time
from tqdm import tqdm

import config
from core.video_processor import extract_frames
from embedders.visual_embedder import VisualEmbedder
from embedders.text_embedder import (
    CaptionGenerator,
    TranscriptExtractor,
    TextEmbedder,
    build_clip_text,
)
from core.indexer import DualIndex


def derive_movie_name(stem: str) -> str:
    """
    'interstellar_01' → 'interstellar'
    'the_dark_knight_02' → 'the_dark_knight'
    Assumes clip ID is the last underscore-separated token if it's a number.
    """
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def build_index() -> DualIndex:
    # video_files is a list of Path objects pointing to .mp4 files in the clips directory
    video_files = sorted(config.CLIPS_DIR.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(
            f"No .mp4 files found in {config.CLIPS_DIR}.\n"
            "Place your clips there and re-run. See README.md for download instructions."
        )

    print(f"\n{'═'*60}")
    print(f"  Scene Retrieval — Index Builder")
    print(f"  Device  : {config.DEVICE}")
    print(f"  Clips   : {len(video_files)}")
    print(f"  Visual  : {config.VISUAL_BACKBONE}")
    print(f"  Caption : {config.CAPTION_MODEL}")
    print(f"  Whisper : {config.WHISPER_SIZE}")
    print(f"{'═'*60}\n")

    # ── Load all models once ──────────────────────────────────────────────────
    visual_embedder    = VisualEmbedder()
    caption_generator  = CaptionGenerator()
    transcript_extractor = TranscriptExtractor()
    text_embedder      = TextEmbedder()

    dual_index = DualIndex()
    t0 = time.time()

    for video_path in tqdm(video_files, desc="Clips", unit="clip"):
        clip_id = video_path.stem
        movie   = derive_movie_name(clip_id)
        print(f"\n── {clip_id} ({movie}) ──")

        # 1. Extract frames
        frames = extract_frames(video_path, num_frames=config.NUM_FRAMES)
        print(f"   frames   : {len(frames)}")

        # 2. Visual embedding
        vis_emb = visual_embedder.embed_frames(frames)
        print(f"   vis_emb  : shape={vis_emb.shape}  norm={float(vis_emb @ vis_emb):.3f}")

        # 3. Caption
        caption = caption_generator.caption_frames(frames)
        print(f"   caption  : {caption[:100]} …")

        # 4. Transcript
        transcript = transcript_extractor.transcribe(str(video_path))
        print(f"   transcript: {(transcript[:80] + ' …') if transcript else '(none)'}")

        # 5. Text embedding
        combined_text = build_clip_text(caption, transcript)
        txt_emb = text_embedder.embed(combined_text)

        # 6. Store
        dual_index.add(
            clip_id=clip_id,
            movie=movie,
            visual_emb=vis_emb,
            text_emb=txt_emb,
            caption=caption,
            transcript=transcript,
            path=str(video_path),
        )

    dual_index.save(config.INDEX_DIR)
    elapsed = time.time() - t0
    print(f"\n✓ Indexed {len(dual_index)} clips in {elapsed:.1f}s")
    return dual_index


if __name__ == "__main__":
    build_index()
