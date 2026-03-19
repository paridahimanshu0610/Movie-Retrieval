"""
video_processor.py — Frame extraction from video clips.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List

from config import NUM_FRAMES


def extract_frames(video_path: str | Path, num_frames: int = NUM_FRAMES) -> List[Image.Image]:
    """
    Uniformly sample `num_frames` frames from a video file.
    Returns a list of PIL Images (RGB).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")

    # Clamp so we never request more frames than exist
    num_frames = min(num_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames: List[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

    cap.release()

    if not frames:
        raise RuntimeError(f"Failed to extract any frames from {video_path}")

    return frames


def video_duration(video_path: str | Path) -> float:
    """Return duration in seconds."""
    cap = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 1
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return total / fps