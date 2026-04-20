"""
pipeline.py

Main indexing pipeline that orchestrates all components.
Processes video clips into searchable scene embeddings.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Generator
from dataclasses import dataclass

# Import our modules
from config import Config, DEFAULT_CONFIG
from indexing.frame_extractor import FrameExtractor, VideoMetadata
from indexing.index_store import SceneIndex, SceneRecord
from models.visual_encoder import InternVideo2Encoder
from models.text_encoder import TextEncoder
from models.transcriber import WhisperTranscriber
from models.captioner import VLMCaptioner, CaptionResult


@dataclass
class ProcessingResult:
    """Result of processing a single clip."""
    success: bool
    scene_id: str
    movie_name: str
    clip_path: str
    error: Optional[str] = None
    processing_time: float = 0.0


class IndexingPipeline:
    """
    Main indexing pipeline for scene-based movie retrieval.
    
    This pipeline processes video clips through:
    1. Frame extraction
    2. Audio transcription (Whisper)
    3. Caption generation (VLM with plot context)
    4. Visual embedding (InternVideo2)
    5. Text embedding (E5)
    
    The result is a searchable index that supports queries about:
    - Visual appearance ("dark underwater scene")
    - Actions/events ("car chase through city")
    - Narrative context ("hero sacrifices himself")
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the indexing pipeline.
        
        Args:
            config: Configuration object. Uses DEFAULT_CONFIG if not provided.
        """
        self.config = config or DEFAULT_CONFIG
        
        # Load plot summaries
        self.plot_data = self._load_plot_data()
        
        # Initialize components (lazy loading)
        self._frame_extractor = None
        self._visual_encoder = None
        self._text_encoder = None
        self._transcriber = None
        self._captioner = None
    
    def _load_plot_data(self) -> Dict:
        """Load plot summaries from JSON file."""
        plot_path = Path(self.config.paths.plot_summaries_path)
        
        if not plot_path.exists():
            print(f"Warning: Plot summaries not found at {plot_path}")
            print("Captions will be generated without plot context.")
            return {}
        
        with open(plot_path) as f:
            data = json.load(f)
        
        print(f"Loaded plot summaries for {len(data)} movies")
        return data
    
    # Lazy initialization of components
    @property
    def frame_extractor(self) -> FrameExtractor:
        """Get frame extractor (lazy initialization)."""
        if self._frame_extractor is None:
            self._frame_extractor = FrameExtractor(
                num_frames=self.config.indexing.max_frames_for_visual,
                sampling_strategy=self.config.indexing.frame_sampling_strategy
            )
        return self._frame_extractor
    
    @property
    def visual_encoder(self) -> InternVideo2Encoder:
        """Get visual encoder (lazy initialization)."""
        if self._visual_encoder is None:
            print("Initializing visual encoder (InternVideo2)...")
            self._visual_encoder = InternVideo2Encoder(
                model_name=self.config.models.visual_encoder_name,
                device=self.config.device,
                use_fp16=self.config.indexing.use_fp16
            )
        return self._visual_encoder
    
    @property
    def text_encoder(self) -> TextEncoder:
        """Get text encoder (lazy initialization)."""
        if self._text_encoder is None:
            print("Initializing text encoder (E5)...")
            self._text_encoder = TextEncoder(
                model_name=self.config.models.text_encoder_name,
                device=self.config.device,
                use_fp16=self.config.indexing.use_fp16
            )
        return self._text_encoder
    
    @property
    def transcriber(self) -> WhisperTranscriber:
        """Get transcriber (lazy initialization)."""
        if self._transcriber is None:
            print("Initializing transcriber (Whisper)...")
            self._transcriber = WhisperTranscriber(
                model_size=self.config.models.whisper_model_size,
                device=self.config.device
            )
        return self._transcriber
    
    @property
    def captioner(self) -> VLMCaptioner:
        """Get captioner (lazy initialization)."""
        if self._captioner is None:
            print("Initializing captioner (VLM)...")
            self._captioner = VLMCaptioner(
                provider=self.config.models.vlm_provider,
                model_name=self.config.models.vlm_model_name,
                max_tokens=self.config.models.vlm_max_tokens
            )
        return self._captioner
    
    def discover_clips(self) -> Generator[tuple, None, None]:
        """
        Discover all video clips to process.
        
        Expects directory structure:
        clips_directory/
            MovieName1/
                scene1.mp4
                scene2.mp4
            MovieName2/
                scene1.mp4
        
        Yields:
            Tuples of (movie_name, clip_path)
        """
        clips_dir = Path(self.config.paths.clips_directory)
        
        if not clips_dir.exists():
            raise ValueError(f"Clips directory not found: {clips_dir}")
        
        # Iterate through movie directories
        for movie_dir in sorted(clips_dir.iterdir()):
            if not movie_dir.is_dir():
                continue
            
            movie_name = movie_dir.name
            
            # Find all video files
            for clip_path in sorted(movie_dir.glob("*.mp4")):
                yield movie_name, clip_path
    
    def get_movie_info(self, movie_name: str) -> Dict:
        """
        Get movie information from plot data.
        
        Args:
            movie_name: Name of the movie (should match key in plot_summaries.json)
            
        Returns:
            Movie info dict with defaults if not found.
        """
        if movie_name in self.plot_data:
            return self.plot_data[movie_name]
        
        # Try case-insensitive match
        for key in self.plot_data:
            if key.lower() == movie_name.lower():
                return self.plot_data[key]
        
        # Return minimal default
        print(f"Warning: No plot data found for '{movie_name}'")
        return {
            "title": movie_name,
            "year": None,
            "genre": [],
            "plot_summary": "Plot information not available.",
            "themes": []
        }
    
    def process_clip(
        self,
        movie_name: str,
        clip_path: Path,
        movie_info: Dict
    ) -> tuple:
        """
        Process a single video clip.
        
        This is the core processing function that:
        1. Extracts frames
        2. Transcribes audio
        3. Generates caption
        4. Creates embeddings
        
        Args:
            movie_name: Name of the movie
            clip_path: Path to the video file
            movie_info: Movie metadata dict
            
        Returns:
            Tuple of (visual_embedding, text_embedding, scene_record)
        """
        clip_name = clip_path.stem
        scene_id = f"{movie_name}_{clip_name}"
        
        # Step 1: Extract frames
        print(f"    Extracting frames...")
        frames, video_meta = self.frame_extractor.extract_frames(str(clip_path))
        
        # Step 2: Transcribe audio
        print(f"    Transcribing audio...")
        try:
            transcription = self.transcriber.transcribe(str(clip_path))
            transcript = transcription.text
        except Exception as e:
            print(f"    Warning: Transcription failed: {e}")
            transcript = None
        
        # Step 3: Generate caption
        print(f"    Generating caption...")
        # Get subset of frames for captioning (VLM doesn't need as many)
        caption_frames = frames[::max(1, len(frames) // self.config.models.vlm_max_frames)]
        caption_frames = caption_frames[:self.config.models.vlm_max_frames]
        
        caption_result = self.captioner.generate_caption(
            frames=caption_frames,
            movie_info=movie_info,
            transcript=transcript if self.config.indexing.include_transcript_in_caption else None,
            video_path=str(clip_path),
        )
        
        # Step 4: Create visual embedding
        print(f"    Creating visual embedding...")
        visual_embedding = self.visual_encoder.encode_video(
            frames,
            normalize=self.config.indexing.normalize_embeddings
        )
        
        # Step 5: Create text embedding
        print(f"    Creating text embedding...")
        embedding_text = caption_result.to_embedding_text(movie_info)
        text_embedding = self.text_encoder.encode_document(
            embedding_text,
            normalize=self.config.indexing.normalize_embeddings
        )
        
        # Create scene record
        scene_record = SceneRecord(
            scene_id=scene_id,
            movie_name=movie_name,
            clip_filename=clip_path.name,
            clip_path=str(clip_path),
            movie_year=movie_info.get("year"),
            movie_genre=movie_info.get("genre", []),
            duration_seconds=video_meta.duration_seconds,
            transcript=transcript[:1000] if transcript else None,  # Truncate
            caption=caption_result.to_dict(),
            embedding_text=embedding_text
        )
        
        return visual_embedding, text_embedding, scene_record
    
    def build_index(self) -> SceneIndex:
        """
        Build the complete scene index.
        
        Processes all clips in the clips directory and creates
        the searchable index.
        
        Returns:
            SceneIndex ready for retrieval.
        """
        print("=" * 60)
        print("SCENE INDEXING PIPELINE")
        print("=" * 60)
        
        index = SceneIndex()
        results: List[ProcessingResult] = []
        
        # Discover all clips
        clips = list(self.discover_clips())
        total_clips = len(clips)
        
        print(f"\nFound {total_clips} clips to process")
        print("-" * 60)
        
        # Process each clip
        for i, (movie_name, clip_path) in enumerate(clips, 1):
            print(f"\n[{i}/{total_clips}] Processing: {movie_name}/{clip_path.name}")
            
            start_time = time.time()
            
            try:
                # Get movie info
                movie_info = self.get_movie_info(movie_name)
                
                # Process the clip
                visual_emb, text_emb, record = self.process_clip(
                    movie_name=movie_name,
                    clip_path=clip_path,
                    movie_info=movie_info
                )
                
                # Add to index
                index.add_scene(visual_emb, text_emb, record)
                
                processing_time = time.time() - start_time
                
                results.append(ProcessingResult(
                    success=True,
                    scene_id=record.scene_id,
                    movie_name=movie_name,
                    clip_path=str(clip_path),
                    processing_time=processing_time
                ))
                
                print(f"    ✓ Completed in {processing_time:.1f}s")
                
            except Exception as e:
                processing_time = time.time() - start_time
                
                results.append(ProcessingResult(
                    success=False,
                    scene_id=f"{movie_name}_{clip_path.stem}",
                    movie_name=movie_name,
                    clip_path=str(clip_path),
                    error=str(e),
                    processing_time=processing_time
                ))
                
                print(f"    ✗ Failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Print summary
        print("\n" + "=" * 60)
        print("INDEXING COMPLETE")
        print("=" * 60)
        
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        total_time = sum(r.processing_time for r in results)
        
        print(f"\nResults:")
        print(f"  Successful: {successful}/{total_clips}")
        print(f"  Failed: {failed}/{total_clips}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average time per clip: {total_time/max(1,total_clips):.1f}s")
        
        if failed > 0:
            print(f"\nFailed clips:")
            for r in results:
                if not r.success:
                    print(f"  - {r.scene_id}: {r.error}")
        
        return index
    
    def run(self) -> SceneIndex:
        """
        Run the complete indexing pipeline.
        
        This is the main entry point. It:
        1. Discovers all clips
        2. Processes each clip
        3. Builds the index
        4. Saves to disk
        
        Returns:
            The completed SceneIndex.
        """
        # Build the index
        index = self.build_index()
        
        # Save to disk
        index.save(self.config.paths.index_directory)
        
        return index


def run_index_pipeline(config_path: Optional[str] = None):
    """
    Convenience function to run the indexing pipeline.
    
    Args:
        config_path: Optional path to YAML config file.
    """
    if config_path:
        config = Config.from_yaml(config_path)
    else:
        config = DEFAULT_CONFIG
    
    pipeline = IndexingPipeline(config)
    index = pipeline.run()
    
    return index
