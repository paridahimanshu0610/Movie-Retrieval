"""
text_encoder.py

Wrapper for text embedding models (E5, BGE).
Used to embed captions for semantic text-based retrieval.
"""

import numpy as np
from typing import List, Union, Optional


class TextEncoder:
    """
    Text encoder for embedding captions and queries.
    
    Uses E5 or BGE models which are state-of-the-art for
    text retrieval tasks. These models produce embeddings
    that capture semantic meaning well.
    
    Key features:
    - Trained specifically for retrieval (not just similarity)
    - Asymmetric encoding: different prefixes for queries vs documents
    - High quality semantic understanding
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/e5-large-v2",
        device: str = "cuda",
        use_fp16: bool = True
    ):
        """
        Initialize text encoder.
        
        Args:
            model_name: HuggingFace model identifier.
                Options:
                - "intfloat/e5-large-v2": 1024-d, best quality
                - "intfloat/e5-base-v2": 768-d, good balance
                - "intfloat/e5-small-v2": 384-d, fastest
                - "BAAI/bge-large-en-v1.5": 1024-d, alternative
                - "BAAI/bge-base-en-v1.5": 768-d, alternative
            device: Device to run on.
            use_fp16: Whether to use half precision.
        """
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"
        
        # Determine model family for correct prompting
        self.is_e5 = "e5" in model_name.lower()
        self.is_bge = "bge" in model_name.lower()
        
        self._load_model()
    
    def _load_model(self):
        """Load the text encoder model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            print(f"Loading text encoder: {self.model_name}")
            
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            # Set precision
            if self.use_fp16:
                self.model.half()
            
            print(f"Text encoder loaded. Embedding dim: {self.embedding_dim}")
            
        except ImportError as e:
            raise ImportError(
                "Please install sentence-transformers: pip install sentence-transformers"
            ) from e
    
    def encode_document(
        self, 
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode a document (caption) for indexing.
        
        E5 and BGE models use different prefixes for documents vs queries.
        This method adds the correct prefix for documents/passages.
        
        Args:
            text: Document text to encode (e.g., scene caption).
            normalize: Whether to L2 normalize the embedding.
            
        Returns:
            Text embedding as numpy array.
        """
        # Add model-specific prefix for documents
        if self.is_e5:
            # E5 uses "passage: " for documents
            prefixed_text = f"passage: {text}"
        elif self.is_bge:
            # BGE doesn't require prefix for documents
            prefixed_text = text
        else:
            prefixed_text = text
        
        embedding = self.model.encode(
            prefixed_text,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        return embedding
    
    def encode_query(
        self, 
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode a query for retrieval.
        
        Uses the query-specific prefix for asymmetric retrieval.
        
        Args:
            text: Query text (e.g., user's scene description).
            normalize: Whether to L2 normalize the embedding.
            
        Returns:
            Query embedding as numpy array.
        """
        # Add model-specific prefix for queries
        if self.is_e5:
            # E5 uses "query: " for queries
            prefixed_text = f"query: {text}"
        elif self.is_bge:
            # BGE uses a specific instruction for queries
            prefixed_text = f"Represent this sentence for searching relevant passages: {text}"
        else:
            prefixed_text = text
        
        embedding = self.model.encode(
            prefixed_text,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        return embedding
    
    def encode_batch(
        self,
        texts: List[str],
        is_query: bool = False,
        normalize: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode multiple texts in batch.
        
        More efficient than encoding one at a time.
        
        Args:
            texts: List of texts to encode.
            is_query: Whether these are queries (True) or documents (False).
            normalize: Whether to normalize embeddings.
            batch_size: Batch size for encoding.
            
        Returns:
            Array of embeddings, shape (len(texts), embedding_dim).
        """
        # Add appropriate prefixes
        if is_query:
            if self.is_e5:
                prefixed_texts = [f"query: {t}" for t in texts]
            elif self.is_bge:
                prefixed_texts = [
                    f"Represent this sentence for searching relevant passages: {t}" 
                    for t in texts
                ]
            else:
                prefixed_texts = texts
        else:
            if self.is_e5:
                prefixed_texts = [f"passage: {t}" for t in texts]
            else:
                prefixed_texts = texts
        
        embeddings = self.model.encode(
            prefixed_texts,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100
        )
        
        return embeddings
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def __repr__(self) -> str:
        return f"TextEncoder(model={self.model_name}, dim={self.embedding_dim})"