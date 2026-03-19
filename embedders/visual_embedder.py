"""
visual_embedder.py — Encode video frames into a single dense visual embedding.

Primary model : InternVideo2-CLIP-1B (temporal transformer, 8 frames)
Fallback model: CLIP ViT-L/14 (mean-pooled over frames)

Both produce L2-normalised embeddings that live in a shared text-visual space,
so a Sentence-BERT query embedding can be compared to them via cosine similarity
at retrieval time.
"""

from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from typing import List

import config


class VisualEmbedder:
    """Wraps either InternVideo2-CLIP or CLIP ViT-L/14."""

    def __init__(self):
        self.device = config.DEVICE
        self._model_type: str = ""
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        if config.VISUAL_BACKBONE == "internvideo2":
            self._try_internvideo2()
        if self._model_type == "":          # not yet loaded → use CLIP fallback
            self._load_clip()

    def _try_internvideo2(self):
        try:
            from transformers import AutoModel, AutoProcessor
            hf_id = "OpenGVLab/InternVideo2-CLIP-1B-224p-f8"
            print(f"[VisualEmbedder] Downloading {hf_id} …")
            self._processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
            self._model = (
                AutoModel.from_pretrained(hf_id, trust_remote_code=True, torch_dtype=torch.float16)
                .to(self.device)
                .eval()
            )
            self._model_type = "internvideo2"
            print("[VisualEmbedder] ✓ InternVideo2-CLIP loaded")
        except Exception as exc:
            print(f"[VisualEmbedder] InternVideo2-CLIP failed ({exc}). Falling back to CLIP ViT-L/14.")

    def _load_clip(self):
        import open_clip
        print("[VisualEmbedder] Loading CLIP ViT-L/14 …")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self._model = model.to(self.device).eval()
        self._preprocess = preprocess
        self._model_type = "clip"
        print("[VisualEmbedder] ✓ CLIP ViT-L/14 loaded")

    # ── Embedding ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def embed_frames(self, frames: List[Image.Image]) -> np.ndarray:
        """
        Returns a 1-D float32 numpy array (L2-normalised).
        """
        if self._model_type == "internvideo2":
            return self._embed_internvideo2(frames)
        return self._embed_clip(frames)

    def _embed_internvideo2(self, frames: List[Image.Image]) -> np.ndarray:
        # InternVideo2 processor expects a list of PIL frames as a "video"
        inputs = self._processor(
            videos=[frames],           # batch of 1 video
            return_tensors="pt",
        ).to(self.device)
        # get_video_features → (1, D)
        feats = self._model.get_video_features(**inputs)
        vec = feats[0].float().cpu().numpy()
        return vec / (np.linalg.norm(vec) + 1e-8)

    def _embed_clip(self, frames: List[Image.Image]) -> np.ndarray:
        # Stack frames → (T, C, H, W)
        tensors = torch.stack([self._preprocess(f) for f in frames]).to(self.device)
        feats = self._model.encode_image(tensors)           # (T, D)
        feats = feats / feats.norm(dim=-1, keepdim=True)   # per-frame L2 norm
        vec = feats.mean(0).float().cpu().numpy()           # temporal mean-pool
        return vec / (np.linalg.norm(vec) + 1e-8)

    @property
    def embedding_dim(self) -> int:
        if self._model_type == "internvideo2":
            return self._model.config.hidden_size
        # CLIP ViT-L/14 → 768
        return 768