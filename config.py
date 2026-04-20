"""
Central configuration for the movie scene retrieval system.
Edit paths and hyperparameters here before running ingest.py or query.py.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "outputs" / "final_clip_data.json"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"
BM25_INDEX_PATH = BASE_DIR / "bm25_index.pkl"

# ── Embedding model (sentence-transformers) ────────────────────────────────────
# bge-small-en-v1.5 is fast and high quality; swap for a larger model if needed.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ── ChromaDB ───────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "movie_clips"

# ── Local LLM (llama-cpp-python) ───────────────────────────────────────────────
# Absolute path to your .gguf model file.
# Recommended: qwen2.5-7b-instruct-q8_0 (better JSON compliance than Llama 3.1 8B)
LLM_MODEL_PATH = "/Users/himanshu/Documents/Projects/policy-and-compliance-reasoning/models/qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf"   # ← set this before running query.py

# Number of model layers to offload to GPU (0 = CPU only).
LLM_N_GPU_LAYERS = -1

# Context window size for the LLM.
LLM_N_CTX = 1000

# ── Field definitions ──────────────────────────────────────────────────────────
# Maps internal field_type labels → JSON keys inside each clip object.
CLIP_FIELDS: dict[str, str] = {
    "visual":    "visual_description",
    "action":    "action",
    "emotional": "emotional_tone",
    "thematic":  "thematic_connection",
    "subjects":  "subjects",
    "description": "description",
}

# Maps internal field_type labels → JSON keys at the movie level.
MOVIE_FIELDS: dict[str, str] = {
    "plot":  "plot_summary",
    "themes": "themes",   # list → joined string at index time
    "tone":  "tone",
}

# All field type labels exposed to the router.
ALL_FIELD_TYPES: list[str] = list(CLIP_FIELDS.keys()) + list(MOVIE_FIELDS.keys())

# ── Retrieval hyperparameters ──────────────────────────────────────────────────
TOP_K = 3        # Number of results returned per retriever call.
RRF_K = 60       # Reciprocal Rank Fusion constant (higher = gentler rank penalty).