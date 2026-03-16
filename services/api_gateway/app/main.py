"""
API Gateway — the single entry point for the recommendation system.

This service orchestrates the full recommendation pipeline:
  1. Receives recommendation requests from clients
  2. Runs the multi-stage ML pipeline (Two-Tower retrieval -> LTR ranking -> Re-ranking)
  3. Returns the final ranked list of recommendations

Pipeline stages:
  - Stage 1: Two-Tower user embedding -> ANN search in Qdrant (200 candidates)
  - Stage 2: LightGBM LTR scores and ranks candidates (32 features)
  - Stage 3: MMR diversity re-ranking + business rules -> top-K results
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.api_gateway.app.routes import health, recommendations

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup: load the full ML recommendation pipeline.
    Shutdown: cleanup resources.
    """
    logger.info("api_gateway.starting", status="loading_pipeline")

    # Try Phase 2 pipeline first (requires trained model artifacts)
    model_dir = Path(__file__).resolve().parents[3] / "models" / "artifacts"
    has_models = (
        (model_dir / "two_tower_best.pt").exists()
        and (model_dir / "two_tower_item_embeddings.npy").exists()
        and (model_dir / "ltr_lightgbm.txt").exists()
    )

    if has_models:
        from services.candidate_svc.app.pipeline_v2 import build_pipeline_v2

        pipeline = build_pipeline_v2()
        app.state.pipeline = pipeline
        app.state.pipeline_version = "v2"
        logger.info("api_gateway.ready", pipeline="v2", model="two_tower+ltr+reranker")
    else:
        from services.candidate_svc.app.pipeline import build_pipeline_from_data

        pipeline, embeddings, id_to_idx = build_pipeline_from_data()
        app.state.pipeline = pipeline
        app.state.embeddings = embeddings
        app.state.id_to_idx = id_to_idx
        app.state.pipeline_version = "v1"
        logger.info("api_gateway.ready", pipeline="v1", model="content_based")

    yield
    logger.info("api_gateway.shutting_down")


app = FastAPI(
    title="FlowMatch RecSys API",
    description="AI Recommendation System — API Gateway",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — configurable origins; defaults to permissive for local dev
import os

_allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False if _allowed_origins == ["*"] else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(health.router, tags=["Health"])
app.include_router(recommendations.router, prefix="/api/v1", tags=["Recommendations"])
