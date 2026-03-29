# Screenplay Retrieval System

A movie identification system: given a natural-language description of a scene,
dialogue, or plot moment, return the most likely matching movie. Built for
CSCE 670 (Information Retrieval).

---

## How it works

Screenplay PDFs are parsed into structured scene JSON, enriched with TMDB
metadata, and indexed using a combination of dense vector search (FAISS),
sparse BM25, and structured inverted indices. At query time, results from
multiple indices are fused using Reciprocal Rank Fusion and aggregated by movie.

---

## Setup

**Requirements:** Python 3.11+

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
TMDB_API_ACCESS_TOKEN=your_token_here
```

---

## Running the pipeline

```bash
# Run all steps
python main.py

# Start from a specific step
python main.py --from convert

# Run a single step
python main.py --only index

# Preview without writing (sync + fetch steps only)
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

1. Drop the PDF/HTML/TXT file into `screenplays/`
2. Run the pipeline from the sync step:

```bash
python main.py --from sync
```

The pipeline will detect the new file, fetch its TMDB metadata, convert and
parse it, and rebuild the indices.

---

## Querying

```bash
# From the project root
python main.py --query "the boy who receives letters from a magic school"
python main.py --query "what are you listening to" --intent dialogue
python main.py --query "a car chase with loud music" --intent scene
python main.py --query "two brothers run a juke joint in the 1930s south" --intent plot
python main.py --query "villain monologue before a fight" --intent dialogue,scene
python main.py --query "..." --mode bm25 --top-k 10
```

**Available intents:**

| Intent | Searches | Best for |
|--------|----------|---------|
| `scene` | heading + action lines | visual descriptions, locations |
| `dialogue` | character: line text | remembered quotes |
| `character` | dialogue grouped by character | character-focused queries |
| `full` | all fields combined | ambiguous or broad queries |
| `plot` | TMDB movie overview | plot summaries, genre/story queries |

---

## Corpus

51 movies — 6,455 scenes — 1,251,415 words

Includes Harry Potter (6 films), Pirates of the Caribbean (4 films), Lord of
the Rings (3 films), Fantastic Beasts (3 films), Tarantino filmography (8
films), and recent releases (2023–2025).

---

## Project structure

```
main.py                      -- run full pipeline
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
  retriever.py               -- query interface + CLI
screenplays/
  manifest.json              -- movie registry
  converted/                 -- normalized plain-text files
output/                      -- scene JSON, metadata JSON, character JSON
indices/                     -- FAISS, BM25, and structured JSON indices
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.
