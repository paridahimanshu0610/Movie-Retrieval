"""
index_store.py

Storage and retrieval for the scene index.
Handles saving/loading embeddings and metadata.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class SceneRecord:
    """
    Complete record for a single scene.
    
    This represents all the information we store about one video clip.
    """
    # Identification
    scene_id: str  # Unique identifier: "{movie_name}_{clip_name}"
    movie_name: str
    clip_filename: str
    clip_path: str
    
    # Movie context
    movie_year: Optional[int]
    movie_genre: List[str]
    
    # Content
    duration_seconds: float
    transcript: Optional[str]
    
    # Caption (structured)
    caption: Dict[str, Any]
    
    # Embedding text (what was actually embedded)
    embedding_text: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneRecord":
        """Create from dictionary."""
        return cls(**data)


class SceneIndex:
    """
    Container for the complete scene index.
    
    Stores:
    - Visual embeddings (from InternVideo2)
    - Text embeddings (from E5)
    - Scene metadata (everything else)
    
    All three are aligned by index: position i in each array
    corresponds to the same scene.
    """
    
    def __init__(
        self,
        visual_embeddings: Optional[np.ndarray] = None,
        text_embeddings: Optional[np.ndarray] = None,
        metadata: Optional[List[SceneRecord]] = None
    ):
        """
        Initialize scene index.
        
        Args:
            visual_embeddings: Array of shape (N, visual_dim)
            text_embeddings: Array of shape (N, text_dim)
            metadata: List of N SceneRecord objects
        """
        self.visual_embeddings = visual_embeddings
        self.text_embeddings = text_embeddings
        self.metadata = metadata or []
    
    def add_scene(
        self,
        visual_embedding: np.ndarray,
        text_embedding: np.ndarray,
        record: SceneRecord
    ):
        """
        Add a scene to the index.
        
        Args:
            visual_embedding: Visual embedding vector (visual_dim,)
            text_embedding: Text embedding vector (text_dim,)
            record: Scene metadata record
        """
        # Initialize arrays if empty
        if self.visual_embeddings is None:
            self.visual_embeddings = visual_embedding.reshape(1, -1)
            self.text_embeddings = text_embedding.reshape(1, -1)
        else:
            self.visual_embeddings = np.vstack([
                self.visual_embeddings, 
                visual_embedding.reshape(1, -1)
            ])
            self.text_embeddings = np.vstack([
                self.text_embeddings, 
                text_embedding.reshape(1, -1)
            ])
        
        self.metadata.append(record)
    
    def __len__(self) -> int:
        """Number of scenes in the index."""
        return len(self.metadata)
    
    def save(self, output_dir: str):
        """
        Save index to disk.
        
        Creates three files:
        - visual_embeddings.npy: Visual embedding array
        - text_embeddings.npy: Text embedding array
        - metadata.json: Scene metadata
        
        Args:
            output_dir: Directory to save index files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings as numpy arrays
        if self.visual_embeddings is not None:
            np.save(
                output_path / "visual_embeddings.npy",
                self.visual_embeddings
            )
        
        if self.text_embeddings is not None:
            np.save(
                output_path / "text_embeddings.npy",
                self.text_embeddings
            )
        
        # Save metadata as JSON
        metadata_dicts = [record.to_dict() for record in self.metadata]
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata_dicts, f, indent=2)
        
        # Save index info
        info = {
            "num_scenes": len(self.metadata),
            "visual_embedding_dim": self.visual_embeddings.shape[1] if self.visual_embeddings is not None else None,
            "text_embedding_dim": self.text_embeddings.shape[1] if self.text_embeddings is not None else None,
        }
        with open(output_path / "index_info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        print(f"Index saved to {output_dir}")
        print(f"  - {len(self.metadata)} scenes")
        print(f"  - Visual embeddings: {self.visual_embeddings.shape}")
        print(f"  - Text embeddings: {self.text_embeddings.shape}")
    
    @classmethod
    def load(cls, index_dir: str) -> "SceneIndex":
        """
        Load index from disk.
        
        Args:
            index_dir: Directory containing index files.
            
        Returns:
            Loaded SceneIndex.
        """
        index_path = Path(index_dir)
        
        # Load embeddings
        visual_embeddings = np.load(index_path / "visual_embeddings.npy")
        text_embeddings = np.load(index_path / "text_embeddings.npy")
        
        # Load metadata
        with open(index_path / "metadata.json") as f:
            metadata_dicts = json.load(f)
        
        metadata = [SceneRecord.from_dict(d) for d in metadata_dicts]
        
        index = cls(
            visual_embeddings=visual_embeddings,
            text_embeddings=text_embeddings,
            metadata=metadata
        )
        
        print(f"Index loaded from {index_dir}")
        print(f"  - {len(index)} scenes")
        
        return index
    
    def get_scene(self, idx: int) -> Dict[str, Any]:
        """
        Get complete information for a scene by index.
        
        Args:
            idx: Scene index.
            
        Returns:
            Dict with embeddings and metadata.
        """
        return {
            "visual_embedding": self.visual_embeddings[idx],
            "text_embedding": self.text_embeddings[idx],
            "metadata": self.metadata[idx]
        }
    
    def get_movie_scenes(self, movie_name: str) -> List[int]:
        """
        Get indices of all scenes from a specific movie.
        
        Args:
            movie_name: Name of the movie.
            
        Returns:
            List of scene indices.
        """
        return [
            i for i, record in enumerate(self.metadata)
            if record.movie_name == movie_name
        ]
    
    def get_all_movies(self) -> List[str]:
        """Get list of all unique movie names in the index."""
        return list(set(record.movie_name for record in self.metadata))