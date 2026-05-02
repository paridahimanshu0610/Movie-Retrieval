# Architecture

CineSearch answers one question: given a natural-language description of a scene, line of dialogue, plot moment, or visual memory, which movie is it from? The system operates across two independent retrieval pipelines — one over parsed screenplays, one over video clips — both accessible through a unified React frontend.

---

## Screenplay pipeline

### Offline indexing

```
PDF / HTML / TXT files
        |
        v
  manifest_sync.py       registers new files -> screenplays/manifest.json
        |
        v
  tmdb_fetch.py          fetches TMDB metadata + Wikipedia plot summaries -> output/{slug}_metadata.json
        |
        v
  convert.py             normalizes to plain-text -> screenplays/converted/{slug}.txt
        |
        v
  batch.py               parses scenes -> output/{slug}_scenes.json
        |
        v
  character_reconcile.py maps screenplay names to TMDB cast -> output/{slug}_characters.json
        |
        v
  index_builder.py       builds all indices -> indices/
```

Run with `cd backend && python main.py`. Individual steps can be run with `--only <step>` or resumed from a checkpoint with `--from <step>`.

### Scene JSON

The parser converts raw screenplay text into scene objects. Each scene is the unit of indexing.

```json
{
  "scene_id":     "baby_driver_scene_003",
  "movie_id":     "339403",
  "scene_number": 3,
  "heading":      "INT./EXT. CAR/PARKING LOT - CONTINUOUS",
  "int_ext":      "INT",
  "location":     "CAR/PARKING LOT",
  "time_of_day":  "CONTINUOUS",
  "characters":   ["GRIFF", "BABY"],
  "word_count":   47,
  "action_lines": [
    "Baby presses play on his iPod.",
    "The music kicks in loud."
  ],
  "dialogue": [
    {
      "character":     "GRIFF",
      "line":          "What are you listening to?",
      "parenthetical": null
    }
  ]
}
```

Character names in screenplays are ALL-CAPS. `character_reconcile.py` maps these to TMDB cast entries (e.g., `GRIFF` → "Jon Bernthal").

### Indices

Five dense+sparse index pairs, each covering a different slice of the screenplay:

| Index | What's indexed | Suited for |
|-------|---------------|------------|
| `scene` | heading + action lines | visual descriptions, locations, physical action |
| `dialogue` | `CHARACTER: line` for every exchange | remembered quotes or lines |
| `character` | all dialogue grouped by character | queries about what a character says or does |
| `full` | all fields concatenated | broad or ambiguous queries |
| `plot` | TMDB overview + Wikipedia summary (one per movie) | plot beats, genres, actor/director names |

Each index is stored as three files: a FAISS flat inner-product index (cosine similarity on normalized vectors), a JSON metadata list in the same order as the FAISS vectors, and a BM25Okapi pickle. Every document carries `scene_id` and `movie_id` as payload.

Six structured JSON indices handle metadata filtering:

| File | Maps |
|------|------|
| `actor_index.json` | actor name → [movie_ids] |
| `director_index.json` | director name → [movie_ids] |
| `genre_index.json` | genre → [movie_ids] |
| `character_index.json` | character name → [{movie_id, actor, scene_count}] |
| `location_index.json` | location → [{movie_id, scene_id}] |
| `registry.json` | movie_id → {title, slug, year, genres, directors, overview, ...} |

Embeddings use `all-MiniLM-L6-v2` via sentence-transformers (384-dim). BM25 tokenization is lowercase alphanumeric only.

### Query classification

Before retrieval, each query is classified by Claude Sonnet 4.5 (via TAMU AI) into a `QueryPlan` using few-shot prompting. The classifier returns a raw JSON object with no parsing hints.

The system recognizes 14 query types:

| Type | Example |
|------|---------|
| `dialogue` | "Someone says I know kung fu" |
| `simple_scene` | "Rooftop fight" |
| `complex_scene` | "Tense argument in the rain about a plan" |
| `detailed_scene` | "Man in suit offers red pill and blue pill" |
| `event_journey` | "Training montage from beginner to master" |
| `plot_level` | "Hacker discovers reality is fake" |
| `similarity` | "Movies like The Matrix" |
| `thematic` | "Movies about redemption" |
| `thematic_scene` | "Scene about betrayal at a dinner table" |
| `filtered` | "90s action movie with a helicopter chase" |
| `multi_criteria` | "Car chase AND rooftop fight" |
| `comparative` | "Like Inception but less confusing" |
| `negation` | "Heist movie but NOT Ocean's Eleven" |
| `multi_aspect` | "A hacker discovers reality is fake and someone says there is no spoon" |

Each type maps to one or more index intents. The classifier also extracts metadata filters (genre, year range), exclude titles (negation), a reference title (similarity/comparative), a query rewrite where appropriate, and sub-queries for `multi_aspect`. If the LLM call fails, the system falls back to a default plan: `plot_level` with `["full"]` intent.

### Retrieval

```
query
  |
  v
intent_classifier.py -> QueryPlan
  |
  v
retriever.py

  1. Build allowed_id set from genre/year constraints and entity (actor/director) matching
  2. Text search: FAISS + BM25 in parallel via ThreadPoolExecutor
       - multi_aspect: each sub-query searched against its own index
       - all other types: query searched against each intent index
       - post-search: results not in allowed_id set are discarded
  3. RRF fusion (k=60): combine all ranked lists into a single fused ranking
  4. Aggregate by movie: sum RRF scores of all scenes from the same film
  5. Return top-k movies with scores and top matching scenes
  |
  v
reranker.py

  6. Single batch LLM call to Claude Sonnet 4.5 with all candidates + matching scenes
  7. LLM returns ranked list of movie IDs; falls back to RRF order on failure
```

**Reciprocal Rank Fusion** is used because FAISS inner-product scores and BM25 scores are not on the same scale. Rank positions are stable across methods; raw scores are not.

**Metadata filtering** builds an allowed-ID set from genre (inverted index, substring match), year (registry scan), and actor/director names (fuzzy match via `difflib.SequenceMatcher`, threshold 0.80, multi-token names only). FAISS and BM25 each fetch an expanded candidate pool when filters are active, and results not in the allowed set are discarded before RRF. A layered fail-open strategy prevents over-constrained queries from returning empty results.

**Multi-aspect queries** split into sub-queries, each searched against its own index independently. A movie that appears in multiple sub-query rankings accumulates RRF contributions from each and floats above movies that only match one aspect.

---

## Video pipeline

### Ingestion

Each clip is processed through two parallel streams:

- **Visual stream:** 12 keyframes sampled and captioned by `Qwen2.5-VL-7B-Instruct`; captions synthesised by `Qwen2.5-7B-Instruct` into four scene-level fields: visual description, subjects, action sequence, and emotional progression.
- **Audio stream:** Audio preprocessed via `ffmpeg`, transcribed by `Whisper-large`, diarized by `pyannote/speaker-diarization-3.1`, and analysed by `Qwen2.5-7B-Instruct` for emotional tone, key actions, corrections/revelations, and contextual details.

The `action` field deliberately combines frame-based and transcript-based signals because frames alone miss dialogue-implied actions and audio alone misses purely visual ones. All fields are merged into a single structured JSON descriptor per clip.

### Indices

Six clip-level fields and three movie-level fields are embedded using `BAAI/bge-small-en-v1.5` and stored in ChromaDB, with each (clip, field) pair as a separate document. A separate BM25 index over dialogue transcripts handles verbatim quote queries.

### Retrieval

```
query
  |
  v
intent_router.py (Qwen2.5-7B-Instruct Q8)
  -> primary field type + optional secondary fields + rewritten query
  |
  v
ChromaDB (primary field WHERE filter) + ChromaDB (secondary fields) + BM25 (if dialogue)
  |
  v
RRF fusion (k=60)
  |
  v
Pairwise reranking (Qwen2.5-7B-Instruct Q8)
  - each pair compared twice with order swapped; winner declared only on both wins
  - stable insertion sort -> final top-3 results
```

---

## API

FastAPI app at `backend/api.py`, runs on port 9000. All FAISS and BM25 indices are loaded into memory at startup and held for the lifetime of the process.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Classify + retrieve + rerank (full screenplay pipeline) |
| `POST` | `/classify` | Return QueryPlan only, no retrieval |
| `POST` | `/search` | Retrieve with an explicit QueryPlan, no LLM classify |
| `GET` | `/movies` | List all movies in corpus |
| `GET` | `/movies/{movie_id}` | Movie detail by TMDB ID |
| `POST` | `/video_query` | Full video retrieval pipeline |
| `POST` | `/reload` | Reload registry and index cache from disk |
| `POST` | `/pipeline/{step}` | Trigger a pipeline step as a background subprocess |

`/search` exists so clients can show the QueryPlan to the user, let them edit it, then submit it directly without re-running the LLM — useful for debugging ranking.

---

## Frontend

Single-page React + TypeScript application (Vite). A mode toggle switches between screenplay and video search. Screenplay results are enriched in parallel with poster art, year, director, and overview via `GET /movies/{id}`. The top result is shown immediately; remaining candidates are available on demand. Video result cards embed the matched YouTube clip at the corresponding timestamp.

---

## Corpus

100 movies — scene, dialogue, character, full, and plot indices.
36 video clips from 10 movies — ChromaDB multi-field + BM25 dialogue index.

---

## Known issues

**Parsing quality.** Some PDFs are image-only scans, CID-encoded, or non-standard FYC versions. These produce zero or near-zero scene counts and are excluded from the manifest manually.

**Short scenes.** Very short scenes (1-2 action lines) produce weak embeddings that match loosely against many queries. No mitigation currently.

**Similarity reranking.** The reranker receives the original query rather than the rewritten one, which can cause the reference film to be spuriously promoted for similarity and comparative queries.

**BM25 refit cost.** BM25Okapi requires the full corpus at fit time. Adding new movies means refitting from scratch; the tokenized corpus is stored alongside the model in the pickle to make this tractable.
