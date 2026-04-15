"""
index_builder.py — Build all retrieval indices from parsed screenplay data.

Dense (FAISS) + Sparse (BM25) indices:
  scene     — heading + action_lines              (visual/action queries)
  dialogue  — CHARACTER: line for every exchange  (quote queries)
  character — dialogue grouped by character       (character-specific queries)
  full      — all fields concatenated             (fallback)
  plot      — TMDB movie overview (one doc/movie) (plot/synopsis queries)

Structured JSON inverted indices:
  actor_index.json      — actor name     -> [movie_ids]
  director_index.json   — director name  -> [movie_ids]
  genre_index.json      — genre          -> [movie_ids]
  character_index.json  — character name -> [{movie_id, actor, scene_count, ...}]
  location_index.json   — location       -> [{movie_id, scene_id}]
  registry.json         — movie_id       -> {title, slug, year, genres, ...}

Each dense/sparse index is stored as:
  {name}.faiss       — FAISS flat inner-product index (cosine on normalized vecs)
  {name}_meta.json   — list of payload dicts in index order
  {name}_bm25.pkl    — serialized BM25Okapi + payloads

Usage:
  python index_builder.py --manifest ../screenplays/manifest.json \\
      --input-dir ../output --index-dir ../indices

  # Sample set (two movies for testing)
  python index_builder.py --sample
"""

import argparse
import json
import logging
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_SLUGS = {"baby_driver", "harry_potter_and_the_chamber_of_secrets"}
EMBED_MODEL  = "all-MiniLM-L6-v2"


def build_scene_text(scene: dict) -> str:
    parts = [scene["heading"]] + scene.get("action_lines", [])
    return " ".join(p for p in parts if p)


def build_dialogue_text(scene: dict) -> str:
    lines = [
        f"{d['character']}: {d['line']}"
        for d in scene.get("dialogue", [])
        if d.get("character") and d.get("line")
    ]
    return " ".join(lines)


def build_character_text(scene: dict) -> str:
    grouped: dict = {}
    for d in scene.get("dialogue", []):
        if d.get("character") and d.get("line"):
            grouped.setdefault(d["character"], []).append(d["line"])
    return " | ".join(f"{c}: {' '.join(ls)}" for c, ls in grouped.items())


def build_full_text(scene: dict) -> str:
    parts = [
        scene.get("heading", ""),
        " ".join(scene.get("action_lines", [])),
        build_dialogue_text(scene),
    ]
    return " ".join(p for p in parts if p)


SCENE_BUILDERS = {
    "scene":     build_scene_text,
    "dialogue":  build_dialogue_text,
    "character": build_character_text,
    "full":      build_full_text,
}


def tokenize(text: str) -> list:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def _build_dense_bm25(texts: list, payloads: list, name: str,
                      index_dir: Path, model: SentenceTransformer) -> None:
    logger.info("  %s: %d documents", name, len(texts))

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    ).astype(np.float32)
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, str(index_dir / f"{name}.faiss"))
    (index_dir / f"{name}_meta.json").write_text(
        json.dumps(payloads, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("    FAISS saved (%d vectors, dim=%d)", faiss_index.ntotal, dim)

    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(index_dir / f"{name}_bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "meta": payloads}, f)
    logger.info("    BM25 saved")


def build_scene_indices(scenes: list, index_dir: Path,
                        model: SentenceTransformer) -> None:
    for index_name, builder in SCENE_BUILDERS.items():
        logger.info("Building scene index: %s", index_name)
        texts, payloads = [], []
        for scene in scenes:
            text = builder(scene).strip()
            if not text:
                continue
            payloads.append({
                "scene_id": scene["scene_id"],
                "movie_id": scene["movie_id"],
                "heading":  scene.get("heading", ""),
                "text":     text,
            })
            texts.append(text)
        _build_dense_bm25(texts, payloads, index_name, index_dir, model)


def build_plot_index(manifest: dict, input_dir: Path, index_dir: Path,
                     model: SentenceTransformer) -> dict:
    """
    Build the plot index from TMDB overview fields (one document per movie).
    Returns the registry dict mapping movie_id to metadata.
    """
    logger.info("Building plot index from TMDB overviews...")
    texts, payloads = [], []
    registry = {}

    for filename, entry in manifest.items():
        slug      = entry["slug"]
        meta_path = input_dir / f"{slug}_metadata.json"
        if not meta_path.exists():
            logger.warning("  No metadata for %s, skipping plot index entry", slug)
            continue

        meta     = json.loads(meta_path.read_text(encoding="utf-8"))
        movie_id = str(meta.get("tmdb_id", entry.get("tmdb_id", "0")))
        title    = meta.get("title", entry.get("title", ""))
        year     = (meta.get("release_date") or "")[:4]
        overview = meta.get("overview", "").strip()

        registry[movie_id] = {
            "title":      title,
            "slug":       slug,
            "year":       year,
            "genres":     meta.get("genres", []),
            "directors":  meta.get("directors", []),
            "overview":   overview,
            "poster_url": meta.get("poster_url", ""),
            "tmdb_url":   meta.get("tmdb_url", ""),
        }

        if overview:
            texts.append(overview)
            payloads.append({
                "movie_id": movie_id,
                "title":    title,
                "slug":     slug,
                "overview": overview,
            })

    _build_dense_bm25(texts, payloads, "plot", index_dir, model)
    return registry


def build_structured_indices(manifest: dict, input_dir: Path,
                              index_dir: Path) -> None:
    logger.info("Building structured indices...")

    actor_index     = defaultdict(list)
    director_index  = defaultdict(list)
    genre_index     = defaultdict(list)
    character_index = defaultdict(list)
    location_index  = defaultdict(list)

    for filename, entry in manifest.items():
        slug        = entry["slug"]
        meta_path   = input_dir / f"{slug}_metadata.json"
        chars_path  = input_dir / f"{slug}_characters.json"
        scenes_path = input_dir / f"{slug}_scenes.json"

        if not meta_path.exists():
            continue

        meta     = json.loads(meta_path.read_text(encoding="utf-8"))
        movie_id = str(meta.get("tmdb_id", entry.get("tmdb_id", "0")))
        title    = meta.get("title", entry.get("title", ""))

        for cast_entry in meta.get("cast", []):
            actor = cast_entry.get("name", "").strip()
            if actor and movie_id not in actor_index[actor.lower()]:
                actor_index[actor.lower()].append(movie_id)

        for director in meta.get("directors", []):
            if director and movie_id not in director_index[director.lower()]:
                director_index[director.lower()].append(movie_id)

        for genre in meta.get("genres", []):
            if genre and movie_id not in genre_index[genre.lower()]:
                genre_index[genre.lower()].append(movie_id)

        if chars_path.exists():
            chars = json.loads(chars_path.read_text(encoding="utf-8"))
            for c in chars:
                if c.get("match_type") == "unmatched":
                    continue
                tmdb_char = c.get("tmdb_character") or c.get("screenplay_name", "")
                key = tmdb_char.lower()
                character_index[key].append({
                    "movie_id":       movie_id,
                    "title":          title,
                    "tmdb_character": tmdb_char,
                    "actor":          c.get("actor", ""),
                    "scene_count":    c.get("scene_count", 0),
                })

        if scenes_path.exists():
            scenes = json.loads(scenes_path.read_text(encoding="utf-8"))
            for scene in scenes:
                loc = scene.get("location", "").strip()
                if loc:
                    location_index[loc.lower()].append({
                        "movie_id": movie_id,
                        "title":    title,
                        "scene_id": scene["scene_id"],
                    })

    def _save(obj, name):
        path = index_dir / name
        path.write_text(json.dumps(dict(obj), indent=2, ensure_ascii=False),
                        encoding="utf-8")
        logger.info("  Saved %s (%d entries)", name, len(obj))

    _save(actor_index,     "actor_index.json")
    _save(director_index,  "director_index.json")
    _save(genre_index,     "genre_index.json")
    _save(character_index, "character_index.json")
    _save(location_index,  "location_index.json")


def load_scenes(input_dir: Path, sample_slugs: set | None = None) -> list:
    all_scenes = []
    for path in sorted(input_dir.glob("*_scenes.json")):
        slug = path.stem.replace("_scenes", "")
        if sample_slugs and slug not in sample_slugs:
            continue
        scenes = json.loads(path.read_text(encoding="utf-8"))
        all_scenes.extend(scenes)
        logger.info("Loaded %s: %d scenes", slug, len(scenes))
    return all_scenes


def main() -> None:
    ap = argparse.ArgumentParser(description="Build all retrieval indices.")
    ap.add_argument("--manifest",   default="../screenplays/manifest.json")
    ap.add_argument("--input-dir",  default="../output")
    ap.add_argument("--index-dir",  default="../indices")
    ap.add_argument("--sample",     action="store_true", help="Use sample set only (2 movies)")
    ap.add_argument("--model",      default=EMBED_MODEL)
    args = ap.parse_args()

    input_dir  = Path(args.input_dir)
    index_dir  = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.sample:
        manifest = {k: v for k, v in manifest.items()
                    if v["slug"] in SAMPLE_SLUGS}

    sample_slugs = set(v["slug"] for v in manifest.values()) if args.sample else None

    logger.info("Loading embedding model: %s", args.model)
    model = SentenceTransformer(args.model)

    scenes = load_scenes(input_dir, sample_slugs)
    if not scenes:
        logger.error("No scenes loaded. Check --input-dir.")
        sys.exit(1)
    logger.info("Total scenes: %d", len(scenes))
    build_scene_indices(scenes, index_dir, model)

    registry = build_plot_index(manifest, input_dir, index_dir, model)
    (index_dir / "registry.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Registry saved (%d movies)", len(registry))

    build_structured_indices(manifest, input_dir, index_dir)

    print(f"\nDone. Indices written to: {index_dir}")
    print(f"  Scene indices (FAISS+BM25): scene, dialogue, character, full, plot")
    print(f"  Structured JSON:            actor, director, genre, character, location, registry")


if __name__ == "__main__":
    main()
