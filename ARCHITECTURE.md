# Architecture

The system answers one question: given a natural-language description of a scene, line of dialogue, or plot moment, which movie is it from? The user remembers something but not the title. The system ranks movies by how well their screenplay content matches the description.

---

## Offline pipeline

```
PDF / HTML / TXT files
        |
        v
  manifest_sync.py       registers new files -> screenplays/manifest.json
        |
        v
  tmdb_fetch.py          fetches metadata, cast, genres -> output/{slug}_metadata.json
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

---

## Scene JSON

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

Character names in screenplays are ALL-CAPS. `character_reconcile.py` maps these to TMDB cast entries (e.g., `GRIFF` → "Jon Bernthal") for the structured indices.

---

## Indices

Five dense+sparse index pairs, each covering a different slice of the screenplay:

| Index | What's indexed | Suited for |
|-------|---------------|------------|
| `scene` | heading + action lines | visual descriptions, locations, physical action |
| `dialogue` | `CHARACTER: line` for every exchange | remembered quotes or lines |
| `character` | all dialogue grouped by character | queries about what a character says or does |
| `full` | all fields concatenated | broad or ambiguous queries |
| `plot` | TMDB overview (one per movie) | plot beats, genres, actor/director names |

Each index is stored as three files: a FAISS flat inner-product index (cosine similarity on normalized vectors), a JSON metadata list in the same order as the FAISS vectors, and a BM25Okapi pickle. Every document in all five indices carries `scene_id` and `movie_id` as payload so results can always be traced back to a movie.

Six structured JSON indices handle queries that don't need embeddings:

| File | Maps |
|------|------|
| `actor_index.json` | actor name → [movie_ids] |
| `director_index.json` | director name → [movie_ids] |
| `genre_index.json` | genre → [movie_ids] |
| `character_index.json` | character name → [{movie_id, actor, scene_count}] |
| `location_index.json` | location → [{movie_id, scene_id}] |
| `registry.json` | movie_id → {title, slug, year, genres, directors, overview, ...} |

Embeddings use `all-MiniLM-L6-v2` via sentence-transformers (384-dim). BM25 tokenization is lowercase alphanumeric only.

---

## Query classification

Before retrieval, each query is classified by an LLM (Claude Sonnet via TAMU AI) into a `QueryPlan`. The classifier runs few-shot prompting with six examples and returns a JSON object — no parsing hints, just raw JSON.

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

Each type maps to one or more index intents. For example, `dialogue` routes to `["dialogue"]`, `complex_scene` to `["scene", "full"]`, `plot_level` to `["plot"]`. The classifier also extracts metadata filters (genre, year range), a query rewrite for similarity/negation/filtered types, and sub-queries for `multi_aspect`.

If the API key is missing or the LLM call fails, the system falls back to a default plan: `plot_level` with `["full"]` intent.

---

## Retrieval

```
query
  |
  v
intent_classifier.py -> QueryPlan
  {
    query_type: "multi_aspect",
    intents: ["plot", "dialogue"],
    filters: {genre: null, year_min: null, year_max: null},
    sub_queries: [
      {intent: "plot",     text: "hacker discovers reality is a simulation"},
      {intent: "dialogue", text: "there is no spoon"}
    ]
  }
  |
  v
retriever.py

  1. Pre-filter: build allowed_id set from genre/year constraints (if any)
  2. Text search:
     - multi_aspect: each sub-query searched against its own index
     - all other types: query searched against each intent index
     - both FAISS and BM25 run in parallel (hybrid mode)
  3. RRF fusion: combine all ranked lists into a single fused ranking
  4. Aggregate by movie: sum RRF scores of all scenes from same movie
  5. Return top-k movies with scores and top matching scenes
```

### Reciprocal Rank Fusion

RRF is used instead of score normalization because FAISS inner-product scores and BM25 scores are not on the same scale. Rank positions are stable across methods; raw scores are not.

```python
def rrf_fuse(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, meta) in enumerate(ranking):
            sid = meta["scene_id"]
            scores[sid] = scores.get(sid, 0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

### Why multi_aspect works

For a query like "a hacker discovers reality is fake and someone says there is no spoon", two sub-queries are searched independently — one against the `plot` index, one against `dialogue`. A movie that appears in both ranked lists gets RRF contributions from two sources and floats above movies that only match one sub-query. The Matrix ends up ranked first not because of a hard rule but because its scenes match both aspects.

---

## API

FastAPI app at `backend/api.py`, runs on port 9000.

The three main search endpoints:

- `POST /classify` — classify query intent, return QueryPlan (no search)
- `POST /search` — search with a provided QueryPlan (no LLM call)
- `POST /query` — classify + search in one call

`/search` exists so the frontend (or any client) can show the QueryPlan to the user, let them edit it, then submit it directly without re-running the LLM. Useful for debugging why a result ranked where it did.

Pipeline steps can be triggered via `POST /pipeline/{step}` — the step runs as a background subprocess and writes output to `pipeline.log`. The registry is loaded at startup and stays in memory; after re-indexing, call `POST /reload` to pick up the new registry without restarting.

---

## Known issues

**Corpus quality.** Some PDFs are image-only scans (no extractable text), CID-encoded (garbled characters), or FYC awards screener versions with non-standard formatting. These are detected by zero or near-zero scene counts and excluded from the manifest manually.

**Ratatouille.** The available PDF is an FYC version with no INT./EXT. scene headings. The parser finds zero scenes. Needs replacement.

**Short scenes.** Very short scenes (1-2 action lines) produce weak embeddings that match almost anything. No mitigation currently.

**BM25 refit cost.** BM25Okapi requires the full corpus at fit time — IDF is computed over all documents. Adding new movies means refitting from scratch. The FAISS index supports `add()` natively. Incremental BM25 is handled by storing the tokenized corpus in the pickle alongside the model object so refitting only requires loading the old corpus and appending.

---

## Corpus

93 movies — scene, dialogue, character, full, and plot indices.
