"""
User Feature Service -- serves user and item features via REST API.

Endpoints:
  GET  /features/user/{user_id}           -- get user features
  GET  /features/item/{item_id}           -- get item features
  POST /features/items                    -- batch get item features
  GET  /features/session/{session_id}     -- get session items
  POST /features/session/{session_id}     -- add item to session
  GET  /features/recent/{user_id}         -- get recent interactions
  GET  /features/stats                    -- store statistics
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.user_feature_svc.app.feature_store import FeatureStore

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: connect to Redis. Shutdown: cleanup."""
    logger.info("user_feature_svc.starting")
    store = FeatureStore(local_mode=True)  # Set to False when Redis is available
    app.state.store = store
    logger.info("user_feature_svc.ready")
    yield
    logger.info("user_feature_svc.shutdown")


app = FastAPI(
    title="User Feature Service",
    description="Serves user and item features from Redis for real-time recommendations",
    version="0.1.0",
    lifespan=lifespan,
)


class ItemBatchRequest(BaseModel):
    item_ids: list[str]


class SessionItemRequest(BaseModel):
    item_id: str


@app.get("/features/user/{user_id}")
async def get_user_features(user_id: str) -> dict:
    """Get pre-computed features for a user."""
    store: FeatureStore = app.state.store
    features = store.get_user_features(user_id)
    if features is None:
        raise HTTPException(status_code=404, detail=f"No features found for user {user_id}")
    return {"user_id": user_id, "features": features}


@app.get("/features/item/{item_id}")
async def get_item_features(item_id: str) -> dict:
    """Get pre-computed features for an item."""
    store: FeatureStore = app.state.store
    features = store.get_item_features(item_id)
    if features is None:
        raise HTTPException(status_code=404, detail=f"No features found for item {item_id}")
    return {"item_id": item_id, "features": features}


@app.post("/features/items")
async def get_item_features_batch(body: ItemBatchRequest) -> dict:
    """Get features for multiple items at once."""
    store: FeatureStore = app.state.store
    results = store.get_item_features_batch(body.item_ids)
    return {"items": results}


@app.get("/features/session/{session_id}")
async def get_session_items(session_id: str) -> dict:
    """Get items viewed in the current session."""
    store: FeatureStore = app.state.store
    items = store.get_session_items(session_id)
    return {"session_id": session_id, "items": items, "count": len(items)}


@app.post("/features/session/{session_id}")
async def add_session_item(session_id: str, body: SessionItemRequest) -> dict:
    """Track an item viewed in this session."""
    store: FeatureStore = app.state.store
    store.add_session_item(session_id, body.item_id)
    return {"status": "added", "session_id": session_id, "item_id": body.item_id}


@app.get("/features/recent/{user_id}")
async def get_recent_interactions(user_id: str, limit: int = 20) -> dict:
    """Get user's most recent interactions."""
    store: FeatureStore = app.state.store
    items = store.get_recent_interactions(user_id, limit=limit)
    return {"user_id": user_id, "recent_items": items, "count": len(items)}


@app.get("/features/stats")
async def get_stats() -> dict:
    """Return feature store statistics."""
    store: FeatureStore = app.state.store
    return store.get_stats()
