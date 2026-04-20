"""
frame_extractor.py

Utilities for extracting frames from video files.
Supports uniform sampling and keyframe-based extraction.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file."""
    path: str
    duration_seconds: float
    total_frames: int
    fps: float
    width: int
    height: int


class FrameExtractor:
    """
    Extracts frames from video files for model processing.
    
    This class handles the conversion from video files to PIL Images
    that can be consumed by vision models like InternVideo2.
    """
    
    def __init__(
        self,
        num_frames: int = 8,
        image_size: Optional[int] = None,
        sampling_strategy: str = "uniform"
    ):
        """
        Initialize the frame extractor.
        
        Args:
            num_frames: Number of frames to extract from each video.
            image_size: If provided, resize frames to this size (square).
            sampling_strategy: How to sample frames - "uniform" or "keyframe".
        """
        self.num_frames = num_frames
        self.image_size = image_size
        self.sampling_strategy = sampling_strategy
    
    def get_video_metadata(self, video_path: str) -> VideoMetadata:
        """
        Extract metadata from a video file without loading all frames.
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            VideoMetadata object with video information.
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            return VideoMetadata(
                path=video_path,
                duration_seconds=duration,
                total_frames=total_frames,
                fps=fps,
                width=width,
                height=height
            )
        finally:
            cap.release()
    
    def extract_frames(
        self, 
        video_path: str,
        num_frames: Optional[int] = None
    ) -> Tuple[List[Image.Image], VideoMetadata]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file.
            num_frames: Override the default number of frames to extract.
            
        Returns:
            Tuple of (list of PIL Images, VideoMetadata).
        """
        num_frames = num_frames or self.num_frames
        
        # Get metadata first
        metadata = self.get_video_metadata(video_path)
        
        # Determine which frames to extract
        if self.sampling_strategy == "uniform":
            frame_indices = self._get_uniform_indices(
                metadata.total_frames, 
                num_frames
            )
        elif self.sampling_strategy == "keyframe":
            frame_indices = self._get_keyframe_indices(
                video_path, 
                metadata.total_frames,
                num_frames
            )
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        # Extract the frames
        frames = self._extract_frames_at_indices(video_path, frame_indices)
        
        return frames, metadata
    
    def _get_uniform_indices(
        self, 
        total_frames: int, 
        num_frames: int
    ) -> List[int]:
        """
        Get uniformly spaced frame indices.
        
        This ensures frames are evenly distributed across the entire video,
        capturing content from beginning, middle, and end.
        
        Args:
            total_frames: Total number of frames in the video.
            num_frames: Number of frames to sample.
            
        Returns:
            List of frame indices.
        """
        if total_frames <= num_frames:
            # Video is shorter than requested frames, return all
            return list(range(total_frames))
        
        # Linspace gives us evenly spaced indices
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        return indices.tolist()
    
    def _get_keyframe_indices(
        self,
        video_path: str,
        total_frames: int,
        num_frames: int
    ) -> List[int]:
        """
        Get frame indices based on scene changes.
        
        Uses simple frame differencing to detect scene changes.
        Falls back to uniform sampling if not enough keyframes found.
        
        Args:
            video_path: Path to the video file.
            total_frames: Total number of frames in the video.
            num_frames: Number of frames to sample.
            
        Returns:
            List of frame indices.
        """
        try:
            # Try to use PySceneDetect if available
            from scenedetect import detect, ContentDetector
            
            scene_list = detect(video_path, ContentDetector())
            
            if len(scene_list) >= num_frames:
                # Get the start frame of each scene
                keyframe_indices = [scene[0].frame_num for scene in scene_list]
                # Sample uniformly from keyframes
                selected_indices = np.linspace(
                    0, len(keyframe_indices) - 1, num_frames, dtype=int
                )
                return [keyframe_indices[i] for i in selected_indices]
            else:
                # Not enough scenes detected, fall back to uniform
                return self._get_uniform_indices(total_frames, num_frames)
                
        except ImportError:
            # PySceneDetect not installed, fall back to uniform sampling
            print("Warning: PySceneDetect not installed, using uniform sampling")
            return self._get_uniform_indices(total_frames, num_frames)
    
    def _extract_frames_at_indices(
        self,
        video_path: str,
        frame_indices: List[int]
    ) -> List[Image.Image]:
        """
        Extract specific frames from a video file.
        
        Args:
            video_path: Path to the video file.
            frame_indices: List of frame indices to extract.
            
        Returns:
            List of PIL Images.
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frames = []
        
        try:
            for idx in frame_indices:
                # Seek to the frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if not ret:
                    print(f"Warning: Could not read frame {idx} from {video_path}")
                    continue
                
                # Convert BGR (OpenCV format) to RGB (PIL format)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                # Resize if specified
                if self.image_size is not None:
                    pil_image = pil_image.resize(
                        (self.image_size, self.image_size),
                        Image.Resampling.LANCZOS
                    )
                
                frames.append(pil_image)
        
        finally:
            cap.release()
        
        return frames
    
    def extract_frames_for_captioning(
        self,
        video_path: str,
        num_frames: int = 8
    ) -> List[Image.Image]:
        """
        Extract frames specifically for VLM captioning.
        
        For captioning, we want representative frames that cover
        the key moments in the scene. We use uniform sampling
        but can adjust based on VLM requirements.
        
        Args:
            video_path: Path to the video file.
            num_frames: Number of frames for the VLM.
            
        Returns:
            List of PIL Images.
        """
        # For captioning, we typically want fewer, well-spaced frames
        frames, _ = self.extract_frames(video_path, num_frames=num_frames)
        return frames