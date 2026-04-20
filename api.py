"""
FastAPI app for querying the movie retrieval index.

Run locally with:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query as FastAPIQuery, Request
from pydantic import BaseModel, Field

from config import LLM_MODEL_PATH, LLM_N_GPU_LAYERS, TOP_K
from pipeline import RetrievalPipeline


def _get_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return int(raw_value)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language scene query.")
    top_k: int = Field(default=TOP_K, ge=1, description="Number of results to return.")


class SearchResult(BaseModel):
    doc_id: str
    clip_id: str
    movie: str
    field_type: str
    score: float
    content: str
    youtube_link: str
    timestamp: str
    description: str
    rrf_score: float | None = None
    matched_fields: list[str] | None = None
    retrieval_rank: int | None = None
    llm_rerank_rank: int | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    use_llm = _get_bool_env("MOVIE_RETRIEVAL_USE_LLM", True)
    llm_model_path = os.getenv("MOVIE_RETRIEVAL_LLM_MODEL_PATH", LLM_MODEL_PATH)
    n_gpu_layers = _get_int_env("MOVIE_RETRIEVAL_LLM_N_GPU_LAYERS", LLM_N_GPU_LAYERS)

    app.state.use_llm = use_llm
    app.state.pipeline = RetrievalPipeline(
        llm_model_path=llm_model_path,
        n_gpu_layers=n_gpu_layers,
        use_llm=use_llm,
    )
    yield
    app.state.pipeline = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="Movie Retrieval API",
        description="Query the indexed movie retrieval pipeline over HTTP.",
        version="1.0.0",
        lifespan=lifespan,
    )

    def _run_query(request: Request, query: str, top_k: int) -> list[dict[str, Any]]:
        normalized_query = query.strip()
        if not normalized_query:
            raise HTTPException(status_code=422, detail="Query must not be empty.")

        pipeline = getattr(request.app.state, "pipeline", None)
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Retrieval pipeline is not loaded.")

        response = pipeline.query(normalized_query, top_k=top_k)
        return response["results"]

    @app.get("/health")
    def healthcheck(request: Request) -> dict[str, Any]:
        return {
            "status": "ok",
            "pipeline_loaded": getattr(request.app.state, "pipeline", None) is not None,
            "use_llm": getattr(request.app.state, "use_llm", None),
        }

    @app.get("/query", response_model=list[SearchResult])
    def query_index_get(
        request: Request,
        query: str = FastAPIQuery(..., min_length=1),
        top_k: int = FastAPIQuery(TOP_K, ge=1),
    ) -> list[dict[str, Any]]:
        return _run_query(request, query=query, top_k=top_k)

    @app.post("/query", response_model=list[SearchResult])
    def query_index_post(
        payload: QueryRequest,
        request: Request,
    ) -> list[dict[str, Any]]:
        return _run_query(request, query=payload.query, top_k=payload.top_k)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("MOVIE_RETRIEVAL_API_HOST", "0.0.0.0"),
        port=_get_int_env("MOVIE_RETRIEVAL_API_PORT", 8000),
        reload=False,
    )
