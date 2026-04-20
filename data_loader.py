"""
Loads the raw clip data JSON file produced by the extraction pipeline.
"""

import json
from pathlib import Path
from typing import Any


def load_clip_data(path: str | Path) -> dict[str, Any]:
    """
    Load movie clip metadata from a JSON file.

    Expected top-level structure:
        {
            "Movie Title": {
                "title": ...,
                "clips": {
                    "clip_id": { ... }
                }
            },
            ...
        }

    Returns the full dict keyed by movie title.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Clip data file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected a top-level JSON object keyed by movie title.")

    print(f"Loaded data for {len(data)} movie(s): {list(data.keys())}")
    return data
