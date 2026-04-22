"""
Intent router powered by a local GGUF model via llama-cpp-python.

Responsibilities
----------------
1. Classify the user query into a PRIMARY field type.
2. Optionally identify SECONDARY field types.
3. Rewrite the query for better embedding retrieval.
4. Flag whether BM25 (dialogue) search should be triggered.

Model recommendation
--------------------
Qwen 2.5 7B Instruct Q8 is preferred over Llama 3.1 8B for this task
because it has stronger structured JSON output compliance.

Fallback
--------
If the LLM call fails or produces unparseable JSON, a keyword-based
heuristic takes over so the pipeline never hard-crashes on bad LLM output.
"""

from __future__ import annotations

import json
import os
import platform
import re
from typing import Any

from dotenv import load_dotenv

from video_config import ALL_FIELD_TYPES, LLM_N_CTX, LLM_N_GPU_LAYERS

# Load variables from .env into environment
load_dotenv()

# ── Prompt template ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a precise query classification assistant for a movie scene retrieval system.
You MUST respond with a single valid JSON object and nothing else — no markdown, no explanation."""

_USER_PROMPT_TEMPLATE = """Classify the following movie scene search query.

Available field types and what they cover:
- visual      : lighting, color palette, cinematography, camera angles, setting appearance
- action      : what physically happens in the scene, movement, events, sequences of events
- emotional   : mood, tone, feelings conveyed to the audience, emotional atmosphere
- thematic    : themes, symbols, deeper meaning, narrative significance
- subjects    : character appearance, costumes, props, physical descriptions of people/objects
- plot        : movie-level plot summary or story description
- tone        : overall movie genre feel (epic, dark, comedic, etc.)

Rules:
- primary_field must be exactly one of the field types above.
- secondary_fields is a list (possibly empty) of other relevant field types.
- rewritten_query must keep the original words intact. Only append missing technical context or expand abbreviations. Do NOT paraphrase, rephrase, or substitute any word. If nothing needs expanding, copy the query exactly. Keep it under 40 words.


User query: "{query}"

Respond ONLY with JSON in this exact schema:
{{
  "primary_field": "<field_type>",
  "secondary_fields": ["<field_type>", ...],
  "rewritten_query": "<expanded query>"
}}"""


def _detect_gpu_backend() -> str:
    """
    Detect the available GPU backend.

    Returns one of: 'metal', 'cuda', 'cpu'
    """
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin" and machine in ("arm64", "x86_64"):
        # Apple Silicon or Intel Mac — Metal is the only GPU path
        return "metal"

    # On Linux / Windows, probe for CUDA via torch if available,
    # otherwise assume CUDA is present when n_gpu_layers > 0.
    try:
        import torch  # optional probe — not a hard dependency
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    # If llama-cpp-python was compiled with CUBLAS and n_gpu_layers != 0,
    # it will use CUDA automatically — trust the caller's config.
    # -1 means "offload all layers", which is also a GPU config.
    return "cuda" if LLM_N_GPU_LAYERS != 0 else "cpu"


def _build_llama_kwargs(n_ctx: int, n_gpu_layers: int, backend: str) -> dict:
    """
    Return the keyword arguments for the Llama constructor that are
    appropriate for the detected backend.

    - Metal  : set n_gpu_layers=-1 (offload everything) and enable
               flash_attn for the slight speed boost Metal supports.
    - CUDA   : pass n_gpu_layers as supplied by config; optionally pin
               memory via tensor splitting when only one device is present.
    - CPU    : force n_gpu_layers=0.
    """
    base = dict(n_ctx=n_ctx, verbose=False)

    if backend == "metal":
        return {
            **base,
            # -1 tells llama.cpp to offload ALL layers to the Metal GPU
            "n_gpu_layers": n_gpu_layers if n_gpu_layers >= 0 else -1,
            # Flash-attention is well-supported on Metal and saves memory
            "flash_attn": True,
        }

    if backend == "cuda":
        return {
            **base,
            "n_gpu_layers": n_gpu_layers,
            # Keep KV cache on the GPU to avoid PCIe round-trips
            "offload_kqv": True,
        }

    # cpu
    return {**base, "n_gpu_layers": 0}


def build_chatml_prompt(system_prompt: str, user_prompt: str) -> str:
    """Format a ChatML-style prompt compatible with Qwen/Llama instruct GGUFs."""
    return (
        "<|im_start|>system\n"
        f"{system_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def load_llama_model(
    model_path: str,
    n_ctx: int = LLM_N_CTX,
    n_gpu_layers: int = LLM_N_GPU_LAYERS,
    *,
    component_name: str = "LocalLLM",
) -> tuple[Any, str, dict]:
    """
    Load a local GGUF model via llama-cpp-python using the shared backend
    selection logic for both routing and reranking.
    """
    if not model_path:
        raise ValueError(
            "LLM_MODEL_PATH is empty. Set it in config.py or pass model_path "
            f"explicitly to {component_name}."
        )

    # Lazy import so the module remains importable for non-LLM workflows.
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is required for local LLM inference. "
            "Install it with:  pip install llama-cpp-python"
        ) from exc

    backend = _detect_gpu_backend()
    llama_kwargs = _build_llama_kwargs(n_ctx, n_gpu_layers, backend)

    print(f"[{component_name}] Loading model from: {model_path}")
    print(f"[{component_name}] GPU backend: {backend}  |  kwargs: {llama_kwargs}")
    llm = Llama(model_path=str(model_path), **llama_kwargs)
    print(f"[{component_name}] Model loaded.")
    return llm, backend, llama_kwargs

# ── Router ─────────────────────────────────────────────────────────────────────

class IntentRouter:
    """
    Wraps a local GGUF LLM to perform intent classification and query
    rewriting.  Falls back to keyword heuristics if the model is
    unavailable or returns malformed output.

    Supports both Apple Metal (Mac) and NVIDIA CUDA GPUs; the correct
    backend is detected automatically at instantiation time.
    """

    def __init__(
        self,
        model_path: str | None = None,
        n_ctx: int = LLM_N_CTX,
        n_gpu_layers: int = LLM_N_GPU_LAYERS,
        llm: Any | None = None,
        backend: str | None = None,
    ) -> None:
        if llm is not None:
            self._llm = llm
            self._backend = backend or _detect_gpu_backend()
            print(f"[IntentRouter] Using shared model instance on backend: {self._backend}")
        else:
            self._llm, self._backend, _ = load_llama_model(
                model_path=model_path or "",
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                component_name="IntentRouter",
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def route(self, query: str) -> dict:
        """
        Classify a user query.

        Returns a dict with keys:
            primary_field   : str
            secondary_fields: list[str]
            rewritten_query : str
        """
        # If query has double quotes around it, then extract the inner text for better parsing by the LLM.
        match = re.match(r'^"(.*)"$', query.strip())
        is_dialogue = False
        if match:
            is_dialogue = True
            query = match.group(1)
        prompt = self._build_prompt(query)
        try:
            raw_output = self._call_llm(prompt)
            result = self._parse_json(raw_output, query)
        except Exception as exc:  # noqa: BLE001
            print(f"[IntentRouter] LLM call failed ({exc}), using keyword fallback.")
            result = keyword_fallback(query)

        # If the original query had double quotes, it's very likely a dialogue search — force that intent.
        print(f"[IntentRouter] Detected dialogue query: {is_dialogue}")
        if is_dialogue:
            result["primary_field"] = "dialogue"
        # Sanitise field names against the allowed list
        result["primary_field"] = _sanitise_field(result.get("primary_field", "action"))
        result["secondary_fields"] = [
            _sanitise_field(f)
            for f in result.get("secondary_fields", [])
            if _sanitise_field(f) != result["primary_field"]
        ]
        return result

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_prompt(self, query: str) -> str:
        """
        Format as a chat prompt compatible with both Llama and Qwen instruct
        models using the standard ChatML / Llama-3 template.
        """
        return build_chatml_prompt(
            _SYSTEM_PROMPT,
            _USER_PROMPT_TEMPLATE.format(query=query),
        )

    def _call_llm(self, prompt: str) -> str:
        response = self._llm(
            prompt,
            max_tokens=256,
            temperature=0.0,          # deterministic output
            stop=["<|im_end|>", "\n\n\n"],
        )
        return response["choices"][0]["text"].strip()

    def _parse_json(self, raw: str, original_query: str) -> dict:
        """Try direct JSON parse, then regex-extract, then fallback."""
        # 1. Direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 2. Extract first {...} block
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # 3. Give up → keyword fallback
        print(f"[IntentRouter] Could not parse JSON from: {raw!r}")
        return keyword_fallback(original_query)


# ── Keyword-based fallback (no LLM required) ──────────────────────────────────

# Maps field type → trigger keywords (all lowercase).
_KEYWORD_MAP: dict[str, list[str]] = {
    "dialogue": [
        "says", "said", "quote", "line", "dialog", "dialogue", "words",
        "speaks", "spoken", "yells", "shouts", "whispers", "tells",
    ],
    "visual": [
        "light", "dark", "color", "colour", "shot", "camera", "looks",
        "visually", "cinematography", "setting", "dim", "bright", "hazy",
        "close-up", "wide", "framing", "palette", "contrast",
    ],
    "emotional": [
        "feel", "emotion", "mood", "tone", "tense", "sad", "happy",
        "scary", "hopeful", "dread", "anxious", "relief", "grief",
        "urgency", "terror", "awe", "uplifting",
    ],
    "subjects": [
        "wearing", "costume", "suit", "dressed", "appearance", "looks like",
        "character", "props", "helmet", "uniform", "outfit", "attire",
        "astronaut", "hair", "face",
    ],
    "action": [
        "happens", "does", "running", "fighting", "jumping", "chasing",
        "escaping", "sequence", "event", "flees", "crashes", "falls",
        "explodes", "launches", "moves",
    ],
    "thematic": [
        "theme", "symbol", "meaning", "represents", "metaphor", "about",
        "deeper", "significance", "motif", "allegory",
    ],
    "plot": [
        "plot", "story", "movie about", "film about", "narrative",
        "synopsis",
    ],
}


def keyword_fallback(query: str) -> dict:
    """
    Simple keyword-match fallback that does NOT require the LLM.
    Returns the same dict shape as IntentRouter.route().
    """
    lower = query.lower()

    primary = "action"  # default
    for field, keywords in _KEYWORD_MAP.items():
        if any(kw in lower for kw in keywords):
            primary = field
            break

    return {
        "primary_field": primary,
        "secondary_fields": [],
        "rewritten_query": query
    }


# ── Utility ────────────────────────────────────────────────────────────────────

def _sanitise_field(field: str) -> str:
    """Return the field if valid, otherwise 'action' as a safe default."""
    return field if field in ALL_FIELD_TYPES + ["dialogue"] else "action"
