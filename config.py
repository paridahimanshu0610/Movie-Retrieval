"""
config.py — Central configuration for the scene retrieval system.
Edit paths and model choices here; everything else reads from this file.
"""

from pathlib import Path
import torch

# ── Device ────────────────────────────────────────────────────────────────────
# M4 Mac → "mps" | CUDA machine → "cuda" | CPU fallback
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
CLIPS_DIR    = PROJECT_ROOT / "clips"     # Drop your .mp4 files here
INDEX_DIR    = PROJECT_ROOT / "index"     # Built index is stored here
CLIPS_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# ── Model choices ─────────────────────────────────────────────────────────────
# Visual backbone.  Options (ordered by accuracy vs. feasibility on 24 GB):
#   "internvideo2"  →  OpenGVLab/InternVideo2-CLIP-1B-224p-f8  (~4 GB, best)
#   "clip_vitl"     →  ViT-L/14 via open_clip                  (~900 MB, fast fallback)
VISUAL_BACKBONE = "clip_vitl"   # set to "clip_vitl" if the HF download fails

# Caption model.  Swap to "llava_video" when you want the full pipeline.
#   "blip2"         →  Salesforce/blip2-opt-2.7b               (~6 GB, easy)
#   "llava_video"   →  lmms-lab/LLaVA-Video-7B-Qwen2           (~15 GB, production)
CAPTION_MODEL = "llava_video"

# Whisper model size: "tiny" | "base" | "small" | "medium"
# "small" is a good tradeoff for testing (≈244 MB)
WHISPER_SIZE = "medium"

# Sentence-BERT model
SBERT_MODEL = "all-mpnet-base-v2"

# Frame sampling
NUM_FRAMES = 20   # sampled uniformly from the clip
NUM_CAPTION_FRAMES = int(0.6 * NUM_FRAMES)  # how many of those frames to caption (≤ NUM_FRAMES)

# Reciprocal Rank Fusion constant
RRF_K = 60