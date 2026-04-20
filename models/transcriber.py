"""
transcriber.py

Wrapper for Whisper speech-to-text transcription.
Extracts dialogue and narration from video clips.
"""

import os
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    """Result of transcribing a video/audio file."""
    text: str
    language: str
    segments: List[Dict]  # List of {start, end, text} dicts
    duration: float


class WhisperTranscriber:
    """
    Speech-to-text transcription using OpenAI's Whisper.
    
    Extracts dialogue and narration from video clips, which is
    important context for understanding what's happening in a scene.
    
    The transcription is used:
    1. As input to the VLM for better captioning
    2. As part of the searchable text content
    """
    
    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16"
    ):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_size: Whisper model size.
                Options: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
                Larger = more accurate but slower.
            device: Device to run on.
            compute_type: Computation type ("float16", "float32", "int8").
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model."""
        try:
            # Try faster-whisper first (much faster with similar quality)
            from faster_whisper import WhisperModel
            
            print(f"Loading Whisper model (faster-whisper): {self.model_size}")
            
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            self.use_faster_whisper = True
            
            print("Whisper loaded successfully (using faster-whisper).")
            
        except ImportError:
            # Fall back to original whisper
            try:
                import whisper
                
                print(f"Loading Whisper model (original): {self.model_size}")
                
                self.model = whisper.load_model(self.model_size, device=self.device)
                self.use_faster_whisper = False
                
                print("Whisper loaded successfully (using original whisper).")
                
            except ImportError as e:
                raise ImportError(
                    "Please install whisper: pip install openai-whisper "
                    "or faster-whisper: pip install faster-whisper"
                ) from e
    
    def _extract_audio(self, video_path: str) -> str:
        """
        Extract audio from video file.
        
        Uses ffmpeg to extract audio as a temporary WAV file.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            Path to temporary audio file.
        """
        # Create temp file for audio
        temp_audio = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        # Extract audio using ffmpeg
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM audio
            "-ar", "16000",  # 16kHz sample rate (Whisper expects this)
            "-ac", "1",  # Mono
            "-y",  # Overwrite
            temp_audio_path
        ]
        
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            # Clean up temp file on error
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            raise RuntimeError(f"Failed to extract audio: {e.stderr}") from e
        except FileNotFoundError:
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg: "
                "https://ffmpeg.org/download.html"
            )
        
        return temp_audio_path
    
    def transcribe(
        self,
        video_path: str,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe speech from a video file.
        
        Args:
            video_path: Path to video file.
            language: Language code (e.g., "en"). None for auto-detection.
            
        Returns:
            TranscriptionResult with text and metadata.
        """
        # Extract audio first
        audio_path = self._extract_audio(video_path)
        
        try:
            if self.use_faster_whisper:
                result = self._transcribe_faster_whisper(audio_path, language)
            else:
                result = self._transcribe_original_whisper(audio_path, language)
            
            return result
            
        finally:
            # Clean up temp audio file
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    
    def _transcribe_faster_whisper(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """Transcribe using faster-whisper."""
        # Run transcription
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,  # Filter out non-speech
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Collect segments
        segment_list = []
        full_text_parts = []
        
        for segment in segments:
            segment_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            full_text_parts.append(segment.text.strip())
        
        full_text = " ".join(full_text_parts)
        
        return TranscriptionResult(
            text=full_text,
            language=info.language,
            segments=segment_list,
            duration=info.duration
        )
    
    def _transcribe_original_whisper(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """Transcribe using original whisper."""
        import whisper
        
        # Run transcription
        result = self.model.transcribe(
            audio_path,
            language=language,
            verbose=False
        )
        
        # Extract segments
        segment_list = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            }
            for seg in result["segments"]
        ]
        
        # Calculate duration from last segment
        duration = segment_list[-1]["end"] if segment_list else 0.0
        
        return TranscriptionResult(
            text=result["text"].strip(),
            language=result["language"],
            segments=segment_list,
            duration=duration
        )
    
    def transcribe_to_text(
        self,
        video_path: str,
        language: Optional[str] = None
    ) -> str:
        """
        Simple method that just returns the transcribed text.
        
        Convenience method for when you don't need segment timestamps.
        
        Args:
            video_path: Path to video file.
            language: Language code or None for auto-detection.
            
        Returns:
            Transcribed text as string.
        """
        result = self.transcribe(video_path, language)
        return result.text
    
    def __repr__(self) -> str:
        return f"WhisperTranscriber(model={self.model_size}, device={self.device})"