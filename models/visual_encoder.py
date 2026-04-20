"""
visual_encoder.py

Wrapper for InternVideo2 video encoder.
Handles video embedding generation with temporal attention.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Union
from pathlib import Path


class InternVideo2Encoder:
    """
    Video encoder using InternVideo2.
    
    InternVideo2 processes multiple frames together with spatiotemporal
    attention, capturing both appearance and motion/action information.
    This is superior to frame-level models + pooling for action understanding.
    
    Key features:
    - Temporal attention across frames (understands motion)
    - Trained on large-scale video-text data (good for retrieval)
    - Produces embeddings aligned with text (same semantic space)
    """
    
    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVideo2-Stage2_1B-224p-f8",
        device: str = "cuda",
        use_fp16: bool = True
    ):
        """
        Initialize InternVideo2 encoder.
        
        Args:
            model_name: HuggingFace model identifier for InternVideo2.
                Options:
                - "OpenGVLab/InternVideo2-Stage2_1B-224p-f4": 1B params, 4 frames
                - "OpenGVLab/InternVideo2-Stage2_1B-224p-f8": 1B params, 8 frames
                - "OpenGVLab/InternVideo2-Stage2_6B-224p-f8": 6B params, 8 frames
            device: Device to run the model on ("cuda" or "cpu").
            use_fp16: Whether to use half precision (faster, less memory).
        """
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"
        
        # Determine number of frames from model name
        if "f4" in model_name:
            self.num_frames = 4
        elif "f8" in model_name:
            self.num_frames = 8
        else:
            self.num_frames = 8  # Default
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the InternVideo2 model and processor."""
        try:
            from transformers import AutoModel, AutoTokenizer, AutoProcessor
            
            print(f"Loading InternVideo2 model: {self.model_name}")
            
            # Determine dtype
            dtype = torch.float16 if self.use_fp16 else torch.float32
            
            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                trust_remote_code=True  # Required for InternVideo2
            ).to(self.device)
            self.model.eval()
            
            # Load tokenizer for text encoding
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load processor for video preprocessing
            # Note: InternVideo2 may have a custom processor
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
            except Exception:
                # Fall back to manual preprocessing if no processor
                self.processor = None
            
            print(f"Model loaded successfully. Using {self.num_frames} frames.")
            
        except ImportError as e:
            raise ImportError(
                "Please install transformers with: pip install transformers"
            ) from e
    
    def preprocess_frames(
        self, 
        frames: List[Image.Image]
    ) -> torch.Tensor:
        """
        Preprocess frames for InternVideo2.
        
        Handles resizing, normalization, and tensor conversion.
        
        Args:
            frames: List of PIL Images (should be self.num_frames images).
            
        Returns:
            Preprocessed tensor of shape (1, num_frames, C, H, W).
        """
        if self.processor is not None:
            # Use the model's processor
            inputs = self.processor(
                videos=frames,
                return_tensors="pt"
            )
            pixel_values = inputs["pixel_values"]
        else:
            # Manual preprocessing
            from torchvision import transforms
            
            # Standard preprocessing for video models
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456,0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Process each frame
            processed_frames = [transform(frame) for frame in frames]
            
            # Stack into tensor: (num_frames, C, H, W)
            pixel_values = torch.stack(processed_frames)
            
            # Add batch dimension: (1, num_frames, C, H, W)
            pixel_values = pixel_values.unsqueeze(0)
        
        # Move to device and set dtype
        dtype = torch.float16 if self.use_fp16 else torch.float32
        pixel_values = pixel_values.to(device=self.device, dtype=dtype)
        
        return pixel_values
    
    def encode_video(
        self, 
        frames: List[Image.Image],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode a video (as frames) into a single embedding vector.
        
        This is the main method for creating visual embeddings.
        InternVideo2 processes all frames together with temporal attention,
        understanding both what appears and what happens.
        
        Args:
            frames: List of PIL Images representing the video.
                Should ideally be self.num_frames images.
            normalize: Whether to L2 normalize the output embedding.
            
        Returns:
            Video embedding as numpy array of shape (embedding_dim,).
        """
        # Ensure we have the right number of frames
        if len(frames) != self.num_frames:
            # Resample to correct number of frames
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        
        # Preprocess
        pixel_values = self.preprocess_frames(frames)
        
        # Encode
        with torch.no_grad():
            # Get video features
            # The exact method name may vary by model version
            if hasattr(self.model, 'encode_video'):
                video_features = self.model.encode_video(pixel_values)
            elif hasattr(self.model, 'get_video_features'):
                video_features = self.model.get_video_features(pixel_values)
            else:
                # Generic forward pass
                outputs = self.model(pixel_values)
                video_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
            
            # Normalize if requested
            if normalize:
                video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        embedding = video_features.cpu().float().numpy().squeeze()
        
        return embedding
    
    def encode_text(
        self, 
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode text for video retrieval.
        
        InternVideo2 is trained with contrastive learning on video-text pairs,
        so text embeddings live in the same space as video embeddings.
        This allows direct comparison between text queries and video embeddings.
        
        Args:
            text: Text query to encode.
            normalize: Whether to L2 normalize the output embedding.
            
        Returns:
            Text embedding as numpy array of shape (embedding_dim,).
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77  # Standard CLIP-style max length
        ).to(self.device)
        
        # Encode
        with torch.no_grad():
            # Get text features
            if hasattr(self.model, 'encode_text'):
                text_features = self.model.encode_text(**inputs)
            elif hasattr(self.model, 'get_text_features'):
                text_features = self.model.get_text_features(**inputs)
            else:
                # Generic approach
                outputs = self.model.get_text_model()(**inputs)
                text_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
            
            # Normalize if requested
            if normalize:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        embedding = text_features.cpu().float().numpy().squeeze()
        
        return embedding
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimension of the embeddings produced by this model."""
        # This varies by model size
        # 1B model: 768, 6B model: typically larger
        if "6B" in self.model_name:
            return 1408  # Approximate, may need adjustment
        else:
            return 768
    
    def __repr__(self) -> str:
        return (
            f"InternVideo2Encoder("
            f"model={self.model_name}, "
            f"frames={self.num_frames}, "
            f"device={self.device})"
        )


# Alternative: LanguageBind implementation for those who prefer it
class LanguageBindVideoEncoder:
    """
    Alternative video encoder using LanguageBind.
    
    LanguageBind offers multimodal alignment (video, audio, image, text)
    in a single embedding space. Good alternative to InternVideo2.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        use_fp16: bool = True
    ):
        """
        Initialize LanguageBind encoder.
        
        Args:
            device: Device to run on.
            use_fp16: Whether to use half precision.
        """
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"
        
        self._load_model()
    
    def _load_model(self):
        """Load LanguageBind model."""
        try:
            from languagebind import LanguageBind, transform_dict, LanguageBindImageTokenizer
            
            print("Loading LanguageBind model...")
            
            # Initialize model with video capability
            self.model = LanguageBind(
                clip_type={'video': 'LanguageBind_Video_FT'},
                cache_dir='./cache'
            ).to(self.device)
            self.model.eval()
            
            # Get transforms
            self.video_transform = transform_dict['video'](self.model)
            
            # Tokenizer for text
            self.tokenizer = LanguageBindImageTokenizer.from_pretrained(
                'LanguageBind/LanguageBind_Image',
                cache_dir='./cache'
            )
            
            print("LanguageBind loaded successfully.")
            
        except ImportError as e:
            raise ImportError(
                "Please install languagebind with: pip install languagebind"
            ) from e
    
    def encode_video_from_path(
        self, 
        video_path: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode video directly from file path.
        
        LanguageBind works directly with video files, handling
        frame extraction and preprocessing internally.
        
        Args:
            video_path: Path to video file.
            normalize: Whether to normalize embedding.
            
        Returns:
            Video embedding as numpy array.
        """
        from languagebind import to_device
        
        # Transform video
        video_data = self.video_transform([video_path])
        video_data = to_device(video_data, self.device)
        
        with torch.no_grad():
            video_features = self.model({'video': video_data})['video']
            
            if normalize:
                video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        
        return video_features.cpu().float().numpy().squeeze()
    
    def encode_text(
        self, 
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """Encode text for retrieval."""
        from languagebind import to_device
        
        text_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True
        )
        text_inputs = to_device(text_inputs, self.device)
        
        with torch.no_grad():
            text_features = self.model({'language': text_inputs})['language']
            
            if normalize:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().float().numpy().squeeze()
    
    @property
    def embedding_dim(self) -> int:
        return 768