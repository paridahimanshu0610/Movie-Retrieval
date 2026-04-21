"""
api.py — FastAPI backend for the screenplay retrieval system.

Endpoints:
  GET  /                    Health check + corpus stats
  POST /classify            LLM intent classification → QueryPlan
  POST /search              Search with an explicit QueryPlan (no LLM call)
  POST /query               Convenience: classify + search in one call
  GET  /movies              List all movies in corpus
  GET  /movies/{movie_id}   Single movie detail by TMDB ID
  POST /pipeline/{step}     Trigger a pipeline step in the background

Run with:
  uvicorn api:app --reload --port 8000

Interactive docs at http://localhost:8000/docs
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT          = Path(__file__).parent.resolve()
RETRIEVAL_DIR = ROOT / "retrieval"
PARSER_DIR    = ROOT / "screenplay_parser"
INDEX_DIR     = ROOT / "indices"

load_dotenv(ROOT / ".env")

# Make retrieval/ importable
sys.path.insert(0, str(RETRIEVAL_DIR))
import retriever as ret
from intent_classifier import classify_query

ret.INDEX_DIR = INDEX_DIR

# Pipeline steps (mirrors main.py STEPS)

_STEPS = {
    "sync": {
        "label":  "Sync manifest with screenplays directory",
        "script": PARSER_DIR / "manifest_sync.py",
        "args":   ["--screenplays-dir", str(ROOT / "screenplays"),
                   "--manifest",        str(ROOT / "screenplays" / "manifest.json")],
        "supports_dry_run":   True,
        "supports_incremental": False,
    },
    "fetch": {
        "label":  "Fetch TMDB metadata",
        "script": PARSER_DIR / "tmdb_fetch.py",
        "args":   ["--manifest",   str(ROOT / "screenplays" / "manifest.json"),
                   "--output-dir", str(ROOT / "output"),
                   "--env",        str(ROOT / ".env")],
        "supports_dry_run":   True,
        "supports_incremental": False,
    },
    "convert": {
        "label":  "Convert screenplay files to plain-text",
        "script": PARSER_DIR / "convert.py",
        "args":   ["--input-dir",  str(ROOT / "screenplays"),
                   "--manifest",   str(ROOT / "screenplays" / "manifest.json"),
                   "--output-dir", str(ROOT / "screenplays" / "converted")],
        "supports_dry_run":   False,
        "supports_incremental": False,
    },
    "parse": {
        "label":  "Parse plain-text into scene JSON",
        "script": PARSER_DIR / "batch.py",
        "args":   ["--converted-dir", str(ROOT / "screenplays" / "converted"),
                   "--manifest",      str(ROOT / "screenplays" / "manifest.json"),
                   "--output-dir",    str(ROOT / "output")],
        "supports_dry_run":   False,
        "supports_incremental": True,
    },
    "reconcile": {
        "label":  "Reconcile screenplay character names with TMDB cast",
        "script": PARSER_DIR / "character_reconcile.py",
        "args":   ["--manifest",   str(ROOT / "screenplays" / "manifest.json"),
                   "--scenes-dir", str(ROOT / "output"),
                   "--output-dir", str(ROOT / "output")],
        "supports_dry_run":   False,
        "supports_incremental": True,
    },
    "index": {
        "label":  "Build retrieval indices",
        "script": RETRIEVAL_DIR / "index_builder.py",
        "args":   ["--manifest",  str(ROOT / "screenplays" / "manifest.json"),
                   "--input-dir", str(ROOT / "output"),
                   "--index-dir", str(INDEX_DIR)],
        "supports_dry_run":   False,
        "supports_incremental": True,
    },
}


app = FastAPI(
    title="Screenplay Retrieval API",
    description="Search 51 movie screenplays by natural-language query.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_registry() -> dict:
    path = INDEX_DIR / "registry.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@app.on_event("startup")
def startup():
    app.state.registry = _load_registry()
    ret.preload_indices()


# Models

class ClassifyRequest(BaseModel):
    query: str

class SubQuery(BaseModel):
    intent: str
    text: str

class Filters(BaseModel):
    genre:    str | None = None
    year_min: int | None = None
    year_max: int | None = None

class QueryPlan(BaseModel):
    query_type:      str
    intents:         list[str]
    filters:         Filters
    exclude_titles:  list[str]        = []
    reference_title: str | None       = None
    rewrite:         str | None       = None
    sub_queries:     list[SubQuery] | None = None

class ClassifyResponse(BaseModel):
    query: str
    plan:  QueryPlan

class SearchRequest(BaseModel):
    query:  str
    plan:   QueryPlan
    mode:   Literal["faiss", "bm25", "hybrid"] = "hybrid"
    top_k:  int = Field(5, ge=1, le=20)

class QueryRequest(BaseModel):
    query:  str
    intent: str | None = None          # comma-separated override, skips LLM
    mode:   Literal["faiss", "bm25", "hybrid"] = "hybrid"
    top_k:  int = Field(5, ge=1, le=20)

class MovieResult(BaseModel):
    rank:      int
    movie_id:  str
    title:     str
    score:     float
    scene_ids: list[str]

class SearchResponse(BaseModel):
    query:        str
    plan:         dict
    entity_names: list[str]
    results:      list[MovieResult]
    total:        int

class MovieSummary(BaseModel):
    movie_id:    str
    title:       str
    year:        str
    directors:   list[str]
    poster_url:  str | None
    tmdb_url:    str | None

class MovieDetail(BaseModel):
    movie_id:    str
    title:       str
    year:        str
    genres:      list[str]
    directors:   list[str]
    overview:    str
    poster_url:  str | None
    tmdb_url:    str | None
    slug:        str
    scene_count: int | None = None

class MovieListResponse(BaseModel):
    movies: list[MovieSummary]
    total:  int

class PipelineRequest(BaseModel):
    dry_run:     bool = False
    incremental: bool = False

class PipelineResponse(BaseModel):
    step:    str
    status:  str
    message: str


# Helpers

def _registry_to_summary(movie_id: str, info: dict) -> MovieSummary:
    return MovieSummary(
        movie_id   = movie_id,
        title      = info.get("title", ""),
        year       = info.get("year", ""),
        directors  = info.get("directors") or [],
        poster_url = info.get("poster_url"),
        tmdb_url   = info.get("tmdb_url"),
    )


def _registry_to_detail(movie_id: str, info: dict) -> MovieDetail:
    return MovieDetail(
        movie_id   = movie_id,
        title      = info.get("title", ""),
        year       = info.get("year", ""),
        genres     = info.get("genres") or [],
        directors  = info.get("directors") or [],
        overview   = info.get("overview", ""),
        poster_url = info.get("poster_url"),
        tmdb_url   = info.get("tmdb_url"),
        slug       = info.get("slug", ""),
        scene_count= info.get("scene_count"),
    )


def _fuzzy_title_to_ids(titles: list[str], registry: dict) -> set:
    ids = set()
    for title in titles:
        tl = title.lower().strip()
        for mid, info in registry.items():
            rt = info.get("title", "").lower().strip()
            if tl == rt or tl in rt or rt in tl:
                ids.add(mid)
    return ids


def _plan_dict_to_retrieve_args(plan: QueryPlan | dict, query: str) -> dict:
    """Convert a QueryPlan (Pydantic or raw dict) into kwargs for ret.retrieve()."""
    if isinstance(plan, QueryPlan):
        f           = plan.filters
        genre       = f.genre if f else None
        yr_min      = f.year_min if f else None
        yr_max      = f.year_max if f else None
        intents     = plan.intents
        rewrite     = plan.rewrite
        exc_titles  = plan.exclude_titles
        raw_sqs     = plan.sub_queries
    else:
        f           = plan.get("filters", {})
        genre       = f.get("genre") if f else None
        yr_min      = f.get("year_min") if f else None
        yr_max      = f.get("year_max") if f else None
        intents     = plan.get("intents", ["full"])
        rewrite     = plan.get("rewrite")
        exc_titles  = plan.get("exclude_titles", [])
        raw_sqs     = plan.get("sub_queries")

    registry   = app.state.registry
    exclude_ids = _fuzzy_title_to_ids(exc_titles, registry)
    year_range  = (yr_min, yr_max) if (yr_min or yr_max) else None
    sub_queries = (
        [{"intent": sq.intent, "text": sq.text} for sq in raw_sqs]
        if raw_sqs and not isinstance(raw_sqs[0], dict)
        else raw_sqs
    )

    return dict(
        query        = rewrite or query,
        intents      = intents,
        genre_filter = genre,
        year_range   = year_range,
        exclude_ids  = exclude_ids,
        sub_queries  = sub_queries,
    )


def _format_results(raw_results: list, top_k: int, registry: dict) -> list[MovieResult]:
    title_map = {mid: info.get("title", f"Movie {mid}") for mid, info in registry.items()}
    out = []
    for rank, r in enumerate(raw_results[:top_k], 1):
        mid       = r["movie_id"]
        scene_ids = [s.get("scene_id", s.get("movie_id", "")) for s in r.get("top_scenes", [])]
        out.append(MovieResult(
            rank      = rank,
            movie_id  = mid,
            title     = title_map.get(mid, f"Movie {mid}"),
            score     = round(r["score"], 4),
            scene_ids = scene_ids,
        ))
    return out


def _run_pipeline_step(step_name: str, dry_run: bool, incremental: bool = False):
    """Background task: run a pipeline step subprocess, log to pipeline.log."""
    step = _STEPS[step_name]
    cmd  = [sys.executable, str(step["script"])] + step["args"]
    if dry_run and step["supports_dry_run"]:
        cmd.append("--dry-run")
    if incremental and step["supports_incremental"]:
        cmd.append("--incremental")
    log_path = ROOT / "pipeline.log"
    with open(log_path, "a", encoding="utf-8") as log:
        label = []
        if dry_run:
            label.append("dry-run")
        if incremental:
            label.append("incremental")
        log.write(f"\n=== {step_name} ({', '.join(label) or 'live'}) ===\n")
        subprocess.run(cmd, cwd=str(step["script"].parent), stdout=log, stderr=log)


# Endpoints

@app.get("/", summary="Health check")
def health():
    registry = app.state.registry
    return {
        "status": "ok",
        "corpus": {
            "movies": len(registry),
            "indices_loaded": (INDEX_DIR / "full.faiss").exists(),
        },
    }


@app.post("/reload", summary="Reload registry and index cache from disk after re-indexing")
def reload_registry():
    app.state.registry = _load_registry()
    ret.clear_index_cache()
    ret.preload_indices()
    return {"status": "ok", "movies": len(app.state.registry)}


@app.post("/classify", response_model=ClassifyResponse, summary="Classify query intent via LLM")
def classify(req: ClassifyRequest):
    plan_dict = classify_query(req.query, env_path=ROOT / ".env")
    # Strip internal keys before returning
    plan_dict.pop("_raw_response", None)
    plan_dict.pop("_mode", None)
    return ClassifyResponse(query=req.query, plan=QueryPlan(**{
        "query_type":      plan_dict.get("query_type", "plot_level"),
        "intents":         plan_dict.get("intents", ["full"]),
        "filters":         plan_dict.get("filters", {}),
        "exclude_titles":  plan_dict.get("exclude_titles", []),
        "reference_title": plan_dict.get("reference_title"),
        "rewrite":         plan_dict.get("rewrite"),
        "sub_queries":     plan_dict.get("sub_queries"),
    }))


@app.post("/search", response_model=SearchResponse, summary="Search with an explicit QueryPlan")
def search(req: SearchRequest):
    kwargs         = _plan_dict_to_retrieve_args(req.plan, req.query)
    kwargs["mode"] = req.mode
    kwargs["top_k"]= req.top_k

    raw_results, entity_names = ret.retrieve(**kwargs)
    results = _format_results(raw_results, req.top_k, app.state.registry)

    return SearchResponse(
        query        = req.query,
        plan         = req.plan.model_dump(),
        entity_names = entity_names,
        results      = results,
        total        = len(results),
    )


@app.post("/query", response_model=SearchResponse, summary="Classify + search in one call")
def query(req: QueryRequest):
    # Build plan: manual intent override or LLM classify
    if req.intent is not None:
        plan_dict = {
            "query_type":      "manual",
            "intents":         [i.strip() for i in req.intent.split(",")],
            "filters":         {"genre": None, "year_min": None, "year_max": None},
            "exclude_titles":  [],
            "reference_title": None,
            "rewrite":         None,
            "sub_queries":     None,
        }
    else:
        plan_dict = classify_query(req.query, env_path=ROOT / ".env")
        plan_dict.pop("_raw_response", None)
        plan_dict.pop("_mode", None)

    kwargs          = _plan_dict_to_retrieve_args(plan_dict, req.query)
    kwargs["mode"]  = req.mode
    kwargs["top_k"] = req.top_k

    raw_results, entity_names = ret.retrieve(**kwargs)
    results = _format_results(raw_results, req.top_k, app.state.registry)

    return SearchResponse(
        query        = req.query,
        plan         = plan_dict,
        entity_names = entity_names,
        results      = results,
        total        = len(results),
    )


@app.get("/movies", response_model=MovieListResponse, summary="List all movies in corpus")
def list_movies():
    registry = app.state.registry
    movies   = [_registry_to_summary(mid, info) for mid, info in registry.items()]
    movies.sort(key=lambda m: m.title)
    return MovieListResponse(movies=movies, total=len(movies))


@app.get("/movies/{movie_id}", response_model=MovieDetail, summary="Get movie detail by TMDB ID")
def get_movie(movie_id: str):
    registry = app.state.registry
    info     = registry.get(movie_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Movie '{movie_id}' not found in corpus.")
    return _registry_to_detail(movie_id, info)


@app.post("/pipeline/{step}", response_model=PipelineResponse, summary="Trigger a pipeline step")
def run_pipeline(step: str, req: PipelineRequest, background_tasks: BackgroundTasks):
    if step not in _STEPS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown step '{step}'. Valid steps: {', '.join(_STEPS)}",
        )
    background_tasks.add_task(_run_pipeline_step, step, req.dry_run, req.incremental)  # noqa
    return PipelineResponse(
        step    = step,
        status  = "started",
        message = f"Step '{step}' is running in the background. Output is appended to pipeline.log.",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=9000,
        reload=True,
        reload_dirs=[str(ROOT)],
        reload_excludes=["*/indices/*", "*/output/*", "*/screenplays/*", "*.log"],
    )
