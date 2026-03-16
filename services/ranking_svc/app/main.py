"""
Ranking Service -- serves the LightGBM LTR model for online candidate scoring.

This service wraps the trained LightGBM model as a REST API:
  POST /rank          -- score and rank a list of candidates for a user
  GET  /rank/health   -- model health check
  GET  /rank/info     -- model metadata (version, features, performance)

In the full pipeline, the flow is:
  candidate_svc (retrieval) -> ranking_svc (scoring) -> reranking_svc (diversity)
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import lightgbm as lgb
import numpy as np
import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = structlog.get_logger()

MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "artifacts"
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "synthetic"


class RankRequest(BaseModel):
    """Request to score and rank candidate items for a user."""

    user_id: str
    candidate_item_ids: list[str] = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=20, ge=1, le=100)


class RankedItem(BaseModel):
    item_id: str
    score: float
    rank: int


class RankResponse(BaseModel):
    user_id: str
    ranked_items: list[RankedItem]
    model_version: str
    num_candidates: int


class ModelInfo(BaseModel):
    model_version: str
    num_features: int
    feature_names: list[str]
    num_trees: int
    best_iteration: int


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load LightGBM model and feature engineer at startup."""
    logger.info("ranking_svc.starting")

    model_path = MODEL_DIR / "ltr_lightgbm.txt"
    if not model_path.exists():
        logger.error("ranking_svc.model_not_found", path=str(model_path))
        raise FileNotFoundError(f"LTR model not found at {model_path}")

    model = lgb.Booster(model_file=str(model_path))
    app.state.model = model
    app.state.model_version = "ltr-v1"

    # Load feature engineer for computing features on the fly
    from services.training_pipeline.app.feature_engineering import FeatureEngineer

    tt_emb_path = str(MODEL_DIR / "two_tower_item_embeddings.npy")
    feature_eng = FeatureEngineer(
        users_path=str(DATA_DIR / "users.jsonl"),
        items_path=str(DATA_DIR / "items.jsonl"),
        interactions_path=str(DATA_DIR / "interactions.jsonl"),
        item_embeddings_path=tt_emb_path if Path(tt_emb_path).exists() else None,
    )
    app.state.feature_eng = feature_eng

    logger.info(
        "ranking_svc.ready",
        num_features=model.num_feature(),
        num_trees=model.num_trees(),
    )
    yield
    logger.info("ranking_svc.shutdown")


app = FastAPI(
    title="Ranking Service",
    description="Scores and ranks candidate items using the LightGBM LTR model",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/rank", response_model=RankResponse)
async def rank_candidates(body: RankRequest) -> RankResponse:
    """Score candidate items and return them ranked by predicted relevance."""
    model: lgb.Booster = app.state.model
    feature_eng = app.state.feature_eng

    # Compute features for each (user, candidate) pair
    pairs = [(body.user_id, item_id) for item_id in body.candidate_item_ids]

    try:
        features = feature_eng.compute_batch_features(pairs)
    except Exception as e:
        logger.error("ranking_svc.feature_error", error=str(e), user_id=body.user_id)
        raise HTTPException(status_code=500, detail=f"Feature computation failed: {e}")

    # Score with LightGBM
    scores = model.predict(features)

    # Sort by score descending and take top_k
    scored_items = list(zip(body.candidate_item_ids, scores))
    scored_items.sort(key=lambda x: x[1], reverse=True)
    top_items = scored_items[: body.top_k]

    ranked = [
        RankedItem(item_id=item_id, score=round(float(score), 6), rank=i + 1)
        for i, (item_id, score) in enumerate(top_items)
    ]

    return RankResponse(
        user_id=body.user_id,
        ranked_items=ranked,
        model_version=app.state.model_version,
        num_candidates=len(body.candidate_item_ids),
    )


@app.get("/rank/health")
async def health() -> dict:
    """Check if the model is loaded and ready."""
    model: lgb.Booster = app.state.model
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": app.state.model_version,
    }


@app.get("/rank/info", response_model=ModelInfo)
async def model_info() -> ModelInfo:
    """Return model metadata."""
    model: lgb.Booster = app.state.model
    return ModelInfo(
        model_version=app.state.model_version,
        num_features=model.num_feature(),
        feature_names=model.feature_name(),
        num_trees=model.num_trees(),
        best_iteration=model.best_iteration,
    )
