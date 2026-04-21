# Screenplay Retrieval System

A movie identification system: given a natural-language description of a scene,
dialogue, or plot moment, return the most likely matching movie. Built for
CSCE 670 (Information Retrieval).

---

## How it works

Screenplay PDFs are parsed into structured scene JSON, enriched with TMDB
metadata, and indexed using dense vector search (FAISS), sparse BM25, and
structured inverted indices. At query time, an LLM classifies the query intent
and routes it to the appropriate indices. Results are fused using Reciprocal
Rank Fusion and aggregated by movie.

---

## Project structure

```
backend/
  main.py                      -- pipeline CLI (run all steps or a single step)
  api.py                       -- FastAPI backend (REST API)
  requirements.txt
  screenplay_parser/
    manifest_sync.py           -- register new screenplay files
    tmdb_fetch.py              -- fetch TMDB metadata
    convert.py                 -- PDF/HTML/TXT -> normalized plain-text
    ingest.py                  -- text extraction utilities
    parse.py                   -- scene boundary detection + content parsing
    serialize.py               -- raw scene dicts -> final JSON schema
    batch.py                   -- batch orchestration over manifest
    character_reconcile.py     -- map screenplay names to TMDB cast
  retrieval/
    index_builder.py           -- build FAISS, BM25, and structured indices
    retriever.py               -- query engine
    intent_classifier.py       -- LLM query classification via TAMU AI
  screenplays/
    manifest.json              -- movie registry (filename -> tmdb_id, slug, title)
    converted/                 -- normalized plain-text files
  output/                      -- scene JSON, metadata JSON, character JSON per movie
  indices/                     -- FAISS, BM25, and structured JSON indices
  .env                         -- API keys (not committed)
frontend/
  src/
    App.tsx                    -- API explorer UI
    App.css
  package.json
  vite.config.ts
```

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+

### Backend

```bash
cd backend
pip install -r requirements.txt
```

Create `backend/.env`:
```
TMDB_API_ACCESS_TOKEN=your_tmdb_token_here
TAMU_AI_CHAT_API_KEY=your_tamu_ai_key_here
```

- **TMDB token** — get one at https://www.themoviedb.org/settings/api (use the "API Read Access Token")
- **TAMU AI key** — required for LLM query classification (Claude Sonnet via chat-api.tamu.ai)

### Frontend

```bash
cd frontend
npm install
```

---

## Running

### Backend API

```bash
python backend/api.py
```

API runs at `http://localhost:9000`. Interactive docs at `http://localhost:9000/docs`.

### Frontend

```bash
cd frontend
npm run dev
```

Frontend runs at `http://localhost:5173`.

---

## Pipeline

Run from the `backend/` directory or prefix commands with `python backend/main.py`.

```bash
cd backend

# Run all steps (full pipeline)
python main.py

# Start from a specific step
python main.py --from convert

# Run a single step
python main.py --only index

# Dry run (preview only — sync and fetch steps)
python main.py --dry-run
```

**Pipeline steps:**

| Step | Script | Output |
|------|--------|--------|
| `sync` | `screenplay_parser/manifest_sync.py` | `screenplays/manifest.json` |
| `fetch` | `screenplay_parser/tmdb_fetch.py` | `output/{slug}_metadata.json` |
| `convert` | `screenplay_parser/convert.py` | `screenplays/converted/{slug}.txt` |
| `parse` | `screenplay_parser/batch.py` | `output/{slug}_scenes.json` |
| `reconcile` | `screenplay_parser/character_reconcile.py` | `output/{slug}_characters.json` |
| `index` | `retrieval/index_builder.py` | `indices/` |

---

## Adding new screenplays

1. Drop the PDF/HTML/TXT file into `backend/screenplays/`
2. Run the pipeline from the sync step:

```bash
cd backend
python main.py --from sync
```

The pipeline will detect the new file, fetch its TMDB metadata, convert and
parse it, and rebuild all indices.

---

## Querying via CLI

```bash
cd backend
python retrieval/retriever.py --query "the boy who receives letters from a magic school"
python retrieval/retriever.py --query "what are you listening to" --intent dialogue
python retrieval/retriever.py --query "a car chase with loud music" --intent scene
python retrieval/retriever.py --query "two brothers run a juke joint in the 1930s south" --intent plot
python retrieval/retriever.py --query "villain monologue before a fight" --intent dialogue,scene
python retrieval/retriever.py --query "..." --mode bm25 --top-k 10
```

**Available intents:**

| Intent | Searches | Best for |
|--------|----------|---------|
| `scene` | heading + action lines | visual descriptions, locations |
| `dialogue` | character speech | remembered quotes |
| `character` | dialogue grouped by character | character-focused queries |
| `full` | all fields combined | ambiguous or broad queries |
| `plot` | TMDB movie overview | plot summaries, genre/theme, actor/director names |

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check — corpus size and index status |
| `GET` | `/movies` | List all movies (title, year, director, poster) |
| `GET` | `/movies/{movie_id}` | Full movie detail by TMDB ID |
| `POST` | `/classify` | LLM intent classification → QueryPlan |
| `POST` | `/query` | Classify + search in one call |
| `POST` | `/search` | Search with an explicit QueryPlan (no LLM) |
| `POST` | `/reload` | Reload registry from disk after re-indexing |
| `POST` | `/pipeline/{step}` | Trigger a pipeline step in the background |

---

## Corpus

93 movies — scene, dialogue, character, full, and plot indices.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design.

---

## Future improvements

**Scene description index**
Generate a natural-language description of each scene during parsing, while the raw screenplay block is still intact. Pass the unstructured text (action lines and dialogue interleaved) to a lightweight LLM (Gemini 2.0 Flash Lite is a strong candidate at ~$0.001 per scene) and store the result as a `description` field in the scene JSON. Build a separate `scene_desc` FAISS + BM25 index from these descriptions. This index would handle complex action queries that the raw scene index struggles with — queries that describe a situation, an emotional context, or a spatial relationship rather than a keyword that appears verbatim in the screenplay.

**Corpus expansion**
Current corpus is 93 movies. Target is 200. Incremental indexing is already implemented — adding a new screenplay only reprocesses that movie and extends the existing indices without a full rebuild.

**Query feedback loop**
Log queries that return zero results or low-confidence scores. Use that data to identify gaps in the corpus (missing movies) or weaknesses in the intent classifier (query types that route to the wrong index).

**Re-ranking with a cross-encoder**
The current pipeline uses bi-encoder embeddings (all-MiniLM-L6-v2) for FAISS and BM25 for keyword matching, fused via RRF. A cross-encoder re-ranker applied to the top-20 results would improve precision for ambiguous queries, at the cost of latency.

**Location index integration**
`location_index.json` is built but not wired into the retriever. Queries that name a specific setting ("a diner," "a rooftop in New York") could use it to pre-filter candidates the same way the genre and year filters work now.
