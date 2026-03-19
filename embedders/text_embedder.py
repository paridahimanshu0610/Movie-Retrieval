"""
text_embedder.py — Build a text embedding for each clip.

Pipeline
────────
1. CaptionGenerator   : BLIP-2 (test)  /  LLaVA-Video (production)
                        → 3 key-frame captions joined into a paragraph
2. TranscriptExtractor: Whisper
                        → spoken dialogue / narration
3. TextEmbedder       : Sentence-BERT  (all-mpnet-base-v2)
                        → dense embedding of  "Caption: … | Transcript: …"

Swap CAPTION_MODEL in config.py to "llava_video" for the full pipeline.
"""

from __future__ import annotations
import torch
import numpy as np
from PIL import Image
from typing import List

import config


# ── 1. Caption Generator ──────────────────────────────────────────────────────

class CaptionGenerator:
    """
    Generates natural-language descriptions from video key frames.

    BLIP-2 (blip2-opt-2.7b) is used for testing; swap to LLaVA-Video-7B
    by setting CAPTION_MODEL = "llava_video" in config.py.
    """

    def __init__(self):
        model_choice = config.CAPTION_MODEL
        if model_choice == "blip2":
            self._load_blip2()
        elif model_choice == "llava_video":
            self._load_llava_video()
        else:
            raise ValueError(f"Unknown CAPTION_MODEL: {model_choice!r}")

    # ── BLIP-2 ────────────────────────────────────────────────────────────────

    def _load_blip2(self):
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        hf_id = "Salesforce/blip2-opt-2.7b"
        print(f"[CaptionGenerator] Loading BLIP-2 ({hf_id}) …")

        # MPS doesn't support bfloat16 well; use float16 on MPS/CUDA, float32 on CPU
        dtype = torch.float16 if config.DEVICE in ("mps", "cuda") else torch.float32

        self._processor = Blip2Processor.from_pretrained(hf_id)
        self._model = (
            Blip2ForConditionalGeneration.from_pretrained(hf_id, torch_dtype=dtype)
            .to(config.DEVICE)
            .eval()
        )
        self._dtype = dtype
        self._gen_fn = self._blip2_caption
        print("[CaptionGenerator] ✓ BLIP-2 loaded")

    @torch.no_grad()
    def _blip2_caption(self, frame: Image.Image) -> str:
        inputs = self._processor(images=frame, return_tensors="pt")
        inputs = {k: v.to(config.DEVICE, self._dtype) if v.is_floating_point() else v.to(config.DEVICE)
                  for k, v in inputs.items()}
        out = self._model.generate(**inputs, max_new_tokens=80)
        return self._processor.decode(out[0], skip_special_tokens=True).strip()

    # ── LLaVA-Video ───────────────────────────────────────────────────────────

    def _load_llava_video(self):
        """
        LLaVA-Video-7B-Qwen2 (lmms-lab/LLaVA-Video-7B-Qwen2).
        Requires ~15 GB RAM.  Set CAPTION_MODEL = "llava_video" in config.py.
        """
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        hf_id = "lmms-lab/LLaVA-Video-7B-Qwen2"
        print(f"[CaptionGenerator] Loading LLaVA-Video ({hf_id}) …")
        self._processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        self._model = (
            LlavaForConditionalGeneration.from_pretrained(
                hf_id, torch_dtype=torch.float16, trust_remote_code=True
            )
            .to(config.DEVICE)
            .eval()
        )
        self._gen_fn = self._llava_caption
        print("[CaptionGenerator] ✓ LLaVA-Video loaded")

    @torch.no_grad()
    def _llava_caption(self, frame: Image.Image) -> str:
        prompt = "Describe what is happening in this video frame in detail."
        inputs = self._processor(text=prompt, images=frame, return_tensors="pt").to(config.DEVICE)
        out = self._model.generate(**inputs, max_new_tokens=200)
        text = self._processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()

    # ── Public API ────────────────────────────────────────────────────────────

    def caption_frames(self, frames: List[Image.Image]) -> str:
        """
        Captions three key frames (first / middle / last) and joins them
        into a multi-sentence paragraph.
        """
        key = [frames[0], frames[len(frames) // 2], frames[-1]]
        parts: List[str] = []
        for i, f in enumerate(key, start=1):
            c = self._gen_fn(f)
            parts.append(f"Frame {i}: {c}")
        return " ".join(parts)


# ── 2. Transcript Extractor ───────────────────────────────────────────────────

class TranscriptExtractor:
    """Uses OpenAI Whisper to transcribe audio from a video file."""

    def __init__(self, size: str = config.WHISPER_SIZE):
        import whisper
        print(f"[TranscriptExtractor] Loading Whisper ({size}) …")
        self._model = whisper.load_model(size)
        print("[TranscriptExtractor] ✓ Whisper loaded")

    def transcribe(self, video_path: str) -> str:
        """Returns the raw transcription string (may be empty)."""
        result = self._model.transcribe(video_path, fp16=False, language="en")
        return result.get("text", "").strip()


# ── 3. Text Embedder (Sentence-BERT) ─────────────────────────────────────────

class TextEmbedder:
    """Embeds arbitrary text with Sentence-BERT (all-mpnet-base-v2)."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        print(f"[TextEmbedder] Loading Sentence-BERT ({config.SBERT_MODEL}) …")
        self._model = SentenceTransformer(config.SBERT_MODEL)
        print("[TextEmbedder] ✓ Sentence-BERT loaded")

    def embed(self, text: str) -> np.ndarray:
        """Returns an L2-normalised float32 numpy vector."""
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Returns (N, D) float32 array, each row L2-normalised."""
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ── 4. Combined text representation ──────────────────────────────────────────

def build_clip_text(caption: str, transcript: str) -> str:
    """
    Combine caption and transcript into a single string fed to Sentence-BERT.
    Transcript is truncated to avoid overwhelming the caption signal.
    """
    transcript = transcript[:400] if transcript else ""
    if transcript:
        return f"Visual description: {caption}. Dialogue: {transcript}"
    return f"Visual description: {caption}"