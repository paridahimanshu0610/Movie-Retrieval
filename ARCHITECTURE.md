# Screenplay Retrieval System — Architecture

## Goal

Given a natural-language description, identify **which movie** the described
scene, dialogue, or plot moment belongs to. The query comes from a user who
remembers something about a movie but not its title.

---

## Pipeline

```
Raw screenplay files (PDF / HTML / TXT)
        |
        v
  [1] manifest_sync.py    -> screenplays/manifest.json  (register new files)
        |
        v
  [2] tmdb_fetch.py       -> output/{slug}_metadata.json  (TMDB metadata + cast)
        |
        v
  [3] convert.py          -> screenplays/converted/{slug}.txt  (normalized text)
        |
        v
  [4] batch.py            -> output/{slug}_scenes.json  (structured scenes)
        |
        v
  [5] character_reconcile.py -> output/{slug}_characters.json  (cast mapping)
        |
        v
  [6] index_builder.py    -> indices/  (FAISS + BM25 + structured JSON)
        |
        v
      retriever.py        -> query -> ranked movie list
```

Run the full pipeline with:
```
python main.py
```

---

## Scene JSON schema

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

---

## Index inventory

| Index | Document unit | Content | Best query type |
|-------|--------------|---------|-----------------|
| `scene` | one scene | heading + action_lines | Visual/action descriptions |
| `dialogue` | one scene | `CHARACTER: line` for every exchange | Remembered quotes |
| `character` | one scene | all dialogue grouped by character | Character-specific queries |
| `full` | one scene | all fields concatenated | Hybrid / ambiguous fallback |
| `plot` | one movie | TMDB overview synopsis | Plot/story beat queries |

Structured JSON inverted indices (no embeddings needed):

| Index | Key | Value |
|-------|-----|-------|
| `actor_index` | actor name | list of movie_ids |
| `director_index` | director name | list of movie_ids |
| `genre_index` | genre | list of movie_ids |
| `character_index` | character name | list of {movie_id, actor, scene_count} |
| `location_index` | location string | list of {movie_id, scene_id} |
| `registry` | movie_id | {title, slug, year, genres, overview, ...} |

Every document in the dense/sparse indices carries `scene_id` and `movie_id` as payload.
Retrieval always resolves back to a ranked movie list.

---

## Query routing

**Option A — Questionnaire (deterministic, no LLM cost)**
```
What best describes what you remember?
  [1] A visual scene or action   -> scene index
  [2] A line of dialogue         -> dialogue index
  [3] A plot moment / story beat -> plot index
  [4] Something a character does -> character index
  [5] A mix / I'm not sure       -> full index + fusion
```

**Option B — LLM intent classification**
```
Prompt: Classify this query into one or more retrieval intents:
        [scene, dialogue, character, plot, hybrid].
        Return JSON: {"intents": [...], "weights": [...]}

Query:  "the part where the villain quotes shakespeare before the final fight"

Response: {"intents": ["dialogue", "scene"], "weights": [0.7, 0.3]}
```

---

## Hybrid retrieval flow

```
user query
    |
    v
query router -> {scene: 0.6, dialogue: 0.4}
    |                 |              |
    |           scene_index    dialogue_index
    |             top-20           top-20
    |                 +---- RRF fusion ----+
    |                             |
    |                   re-rank by scene_id overlap
    |                   (same scene in multiple index results = strong signal)
    |                             |
    v                             v
group hits by movie_id  ->  score = sum of RRF scores
    |
    v
ranked movie list  ("This sounds like Harry Potter")
```

### Reciprocal Rank Fusion

```python
def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

---

## Retrieval challenges

| Challenge | Mitigation |
|-----------|-----------|
| Paraphrase gap ("magic school letters" vs "Hogwarts owls") | Dense embeddings; fine-tuned model preferred over BM25 alone |
| Imbalanced corpus (Interstellar 367 scenes vs Death Proof 28) | RRF aggregation rewards multiple hits from same movie |
| Generic scenes (car chase, argument at dinner) | Require multiple top hits from same movie before committing |
| Multi-scene queries ("the part where X then Y happens") | Chunk query, retrieve independently, intersect by movie |

---

## File map

```
main.py                                 -- run full pipeline (all steps)

screenplay_parser/
  manifest_sync.py    -- scan screenplays dir, add new files to manifest.json
  tmdb_fetch.py       -- fetch TMDB metadata (overview, cast, genres, directors)
  convert.py          -- PDF/HTML/TXT -> normalized plain-text
  ingest.py           -- lower-level text extraction (used by batch.py directly)
  parse.py            -- ScreenPy scene detection + custom content parser
  serialize.py        -- raw scene dicts -> final JSON schema
  batch.py            -- CLI: runs convert -> parse -> serialize over manifest
  character_reconcile.py -- map screenplay ALL-CAPS names to TMDB cast entries

retrieval/
  index_builder.py    -- build FAISS, BM25, and structured JSON indices
  retriever.py        -- query indices, RRF fusion, return ranked movie list

screenplays/
  manifest.json             -- {filename: {tmdb_id, title, slug}}
  converted/{slug}.txt      -- normalized plain-text output of convert.py

output/
  {slug}_scenes.json        -- structured scene objects
  {slug}_metadata.json      -- TMDB metadata (overview, cast, genres, directors)
  {slug}_characters.json    -- screenplay name -> TMDB cast reconciliation

indices/
  scene.faiss / scene_meta.json / scene_bm25.pkl
  dialogue.faiss / dialogue_meta.json / dialogue_bm25.pkl
  character.faiss / character_meta.json / character_bm25.pkl
  full.faiss / full_meta.json / full_bm25.pkl
  plot.faiss / plot_meta.json / plot_bm25.pkl
  actor_index.json / director_index.json / genre_index.json
  character_index.json / location_index.json / registry.json
```

---

## Current corpus

51 movies — 6,455 scenes — 1,251,415 words — 44,643 dialogue lines

Includes: Harry Potter series (6 films), Pirates of the Caribbean (4 films),
Lord of the Rings (3 films), Fantastic Beasts (3 films), Tarantino filmography
(8 films), and recent releases (2023-2025).
