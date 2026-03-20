"""
text_embedder.py — Build a text embedding for each clip.

Pipeline
────────
1. CaptionGenerator   : BLIP-2 (test) / LLaVA-Video (production)
                        → description from sampled frames or whole-video reasoning
2. TranscriptExtractor: Whisper
                        → spoken dialogue / narration
3. TextEmbedder       : Sentence-BERT (all-mpnet-base-v2)
                        → dense embedding of "Caption: ... | Transcript: ..."

Swap CAPTION_MODEL in config.py to "llava_video" for the full pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch

import config


# ── 1. Caption Generator ──────────────────────────────────────────────────────

class CaptionGenerator:
    """
    Generates natural-language descriptions for each clip.

    BLIP-2 captions a subset of sampled frames.
    LLaVA-Video reasons over the clip as a video and produces one description.
    """

    def __init__(self):
        self._model_choice = config.CAPTION_MODEL
        if self._model_choice == "blip2":
            self._load_blip2()
        elif self._model_choice == "llava_video":
            self._load_llava_video()
        else:
            raise ValueError(f"Unknown CAPTION_MODEL: {self._model_choice!r}")

    # ── BLIP-2 ────────────────────────────────────────────────────────────────

    def _load_blip2(self):
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        hf_id = "Salesforce/blip2-opt-2.7b"
        print(f"[CaptionGenerator] Loading BLIP-2 ({hf_id}) ...")

        # MPS doesn't support bfloat16 well; use float16 on MPS/CUDA, float32 on CPU.
        self._dtype = torch.float16 if config.DEVICE in ("mps", "cuda") else torch.float32
        self._processor = Blip2Processor.from_pretrained(hf_id)
        self._model = (
            Blip2ForConditionalGeneration.from_pretrained(hf_id, torch_dtype=self._dtype)
            .to(config.DEVICE)
            .eval()
        )
        print("[CaptionGenerator] ✓ BLIP-2 loaded")

    @torch.no_grad()
    def _blip2_caption(self, frame: Image.Image) -> str:
        inputs = self._processor(images=frame, return_tensors="pt")
        inputs = {
            k: v.to(config.DEVICE, self._dtype) if v.is_floating_point() else v.to(config.DEVICE)
            for k, v in inputs.items()
        }
        out = self._model.generate(**inputs, max_new_tokens=200)
        return self._processor.decode(out[0], skip_special_tokens=True).strip()

    def _caption_selected_frames(self, frames: List[Image.Image]) -> str:
        if not frames:
            raise ValueError("caption(...) expected at least one frame.")

        n = min(config.NUM_CAPTION_FRAMES, len(frames))
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        selected_frames = [frames[i] for i in indices]

        parts: List[str] = []
        for i, frame in enumerate(selected_frames, start=1):
            parts.append(f"Frame {i}: {self._blip2_caption(frame)}")
        return " ".join(parts)

    # ── LLaVA-Video ───────────────────────────────────────────────────────────

    def _load_llava_video(self):
        """
        LLaVA-NeXT-Video-7B-hf (llava-hf/LLaVA-NeXT-Video-7B-hf).
        Uses the Transformers-native video model.
        """
        import av
        from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

        hf_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
        print(f"[CaptionGenerator] Loading LLaVA-NeXT-Video ({hf_id}) ...")

        self._av = av
        self._dtype = torch.float16 if config.DEVICE in ("mps", "cuda") else torch.float32
        self._processor = LlavaNextVideoProcessor.from_pretrained(hf_id)
        self._model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            hf_id,
            torch_dtype=self._dtype,
            low_cpu_mem_usage=True,
        ).to(config.DEVICE)
        self._model.eval()
        print("[CaptionGenerator] ✓ LLaVA-NeXT-Video loaded")

    def _load_video_for_llava(
        self,
        video_path: str | Path,
        num_frames: int,
    ) -> tuple[np.ndarray, str, float]:
        container = self._av.open(str(video_path))
        stream = container.streams.video[0]
        total_frames = stream.frames
        if total_frames <= 0:
            frames = [frame for frame in container.decode(video=0)]
            total_frames = len(frames)
            if total_frames == 0:
                raise ValueError(f"Video {video_path} contains no decodable frames.")
            decoded = [frame.to_ndarray(format="rgb24") for frame in frames]
            fps = float(stream.average_rate) if stream.average_rate else 1.0
            container.close()
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frame_indices = np.unique(frame_indices)   # remove duplicates if clip is short
            clip = np.stack([decoded[i] for i in frame_indices])
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frame_indices = np.unique(frame_indices)   # remove duplicates if clip is short
            frame_set = set(int(i) for i in frame_indices.tolist())
            decoded = []
            container.seek(0)
            for i, frame in enumerate(container.decode(video=0)):
                if i > int(frame_indices[-1]):
                    break
                if i in frame_set:
                    decoded.append(frame.to_ndarray(format="rgb24"))
            fps = float(stream.average_rate) if stream.average_rate else 1.0
            container.close()
            clip = np.stack(decoded)

        video_time = total_frames / fps
        frame_time = ",".join(f"{i / fps:.2f}s" for i in frame_indices)
        return clip, frame_time, video_time

    @torch.no_grad()
    def _llava_caption(self, video_path: str | Path) -> str:
        clip, frame_time, video_time = self._load_video_for_llava(
            video_path,
            num_frames=max(8, config.NUM_CAPTION_FRAMES),
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"The video lasts for {video_time:.2f} seconds. "
                            f"{len(clip)} uniformly sampled frames are provided at {frame_time}. "
                            "Please describe this video in detail."
                        ),
                    },
                    {"type": "video"},
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self._processor(
            text=prompt,
            videos=clip,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        output_ids = self._model.generate(
            **inputs,
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
        )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # ── Public API ────────────────────────────────────────────────────────────

    def caption(self, frames: List[Image.Image], video_path: str | Path | None = None) -> str:
        if self._model_choice == "blip2":
            return self._caption_selected_frames(frames)

        if video_path is None:
            raise ValueError("caption(..., video_path=...) is required for llava_video.")
        return self._llava_caption(video_path)


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
    transcript = transcript[:1000] if transcript else ""
    if transcript:
        return f"Visual description: {caption}. Dialogue: {transcript}"
    return f"Visual description: {caption}"
