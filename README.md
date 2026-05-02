# CineSearch

A natural-language movie identification system built for CSCE 670 (Information Retrieval) at Texas A&M University. Given a free-form description of a scene, line of dialogue, plot moment, or visual memory, the system returns the most likely matching movie.

The system operates across two independent retrieval pipelines: a **screenplay pipeline** over 100 parsed feature films, and a **video pipeline** over 36 YouTube clips from 10 movies.

---

## How it works

**Screenplay pipeline:** Screenplay PDFs are parsed into structured scene JSON, enriched with TMDB metadata and Wikipedia plot summaries, and indexed using dense vector search (FAISS) and sparse BM25 across five index types. At query time, an LLM classifies the query intent and routes it to the appropriate indices. Results are fused using Reciprocal Rank Fusion, aggregated by movie, and reranked by an LLM in a single batch call.

**Video pipeline:** Short movie clips are processed through keyframe captioning (Qwen2.5-VL-7B-Instruct) and speech transcription (Whisper-large), producing multi-field descriptors stored in ChromaDB. Queries are routed by a local LLM to the appropriate field type, retrieved via hybrid search, and reranked using pairwise LLM comparison.

---

## Project structure

```
backend/
  api.py                       -- FastAPI backend (REST API, port 9000)
  main.py                      -- pipeline CLI
  requirements.txt
  screenplay_parser/
    manifest_sync.py           -- register new screenplay files
    tmdb_fetch.py              -- fetch TMDB + Wikipedia metadata
    convert.py                 -- PDF/HTML/TXT -> normalized plain-text
    parse.py                   -- scene boundary detection + content parsing
    batch.py                   -- batch orchestration over manifest
    character_reconcile.py     -- map screenplay names to TMDB cast
  retrieval/
    index_builder.py           -- build FAISS, BM25, and structured indices
    retriever.py               -- query engine
    intent_classifier.py       -- LLM query classification via TAMU AI
    reranker.py                -- LLM batch reranker via TAMU AI
  video_router/
    intent_router.py           -- local LLM intent routing for video queries
  video_retriever/
    video_retrieval_pipeline.py -- full video retrieval pipeline
  video_config.py              -- video pipeline configuration
  screenplays/
    manifest.json              -- movie registry
    converted/                 -- normalized plain-text files
  output/                      -- scene JSON, metadata JSON per movie
  indices/                     -- FAISS, BM25, and structured JSON indices
  .env                         -- API keys (not committed)
frontend/
  src/
    search-page.tsx            -- main search UI (screenplay + video modes)
    App.tsx
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
- **TAMU AI key** — required for LLM query classification and reranking (Claude Sonnet via chat-api.tamu.ai)

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

Run from the `backend/` directory.

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

---

## Querying via CLI

```bash
cd backend
python retrieval/retriever.py --query "the boy who receives letters from a magic school"
python retrieval/retriever.py --query "what are you listening to" --intent dialogue
python retrieval/retriever.py --query "a car chase with loud music" --intent scene
python retrieval/retriever.py --query "two brothers run a juke joint in the 1930s south" --intent plot
python retrieval/retriever.py --query "..." --mode bm25 --top-k 10
```

**Available intents:**

| Intent | Searches | Best for |
|--------|----------|---------|
| `scene` | heading + action lines | visual descriptions, locations |
| `dialogue` | character speech | remembered quotes |
| `character` | dialogue grouped by character | character-focused queries |
| `full` | all fields combined | ambiguous or broad queries |
| `plot` | TMDB overview + Wikipedia summary | plot summaries, genre/theme, actor/director names |

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check — corpus size and index status |
| `GET` | `/movies` | List all movies (title, year, director, poster) |
| `GET` | `/movies/{movie_id}` | Full movie detail by TMDB ID |
| `POST` | `/classify` | LLM intent classification → QueryPlan |
| `POST` | `/query` | Classify + retrieve + rerank in one call |
| `POST` | `/search` | Search with an explicit QueryPlan (no LLM classify) |
| `POST` | `/reload` | Reload registry and indices from disk after re-indexing |
| `POST` | `/pipeline/{step}` | Trigger a pipeline step in the background |
| `POST` | `/video_query` | Query the video retrieval pipeline |

---

## Corpus

100 movies — scene, dialogue, character, full, and plot indices.
36 video clips from 10 movies — multi-field ChromaDB + BM25 dialogue index.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design.
