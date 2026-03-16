"""
Recommendation endpoints -- the core API surface.

POST /api/v1/recommend
  Takes a user_id + optional context and returns a ranked list of item recommendations.

Supports two pipeline versions:
  - V2 (Phase 2): Two-Tower retrieval -> LightGBM LTR -> MMR re-ranking
  - V1 (Phase 1 fallback): Content-based embedding retrieval
"""

from fastapi import APIRouter, Request

from shared.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendedItem,
)

router = APIRouter()


@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request: Request,
    body: RecommendationRequest,
) -> RecommendationResponse:
    """
    Get personalized recommendations for a user.

    Pipeline V2 (Two-Tower + LTR + Re-ranking):
      - If user_id provided -> full ML pipeline (retrieval + ranking + re-ranking)
      - If session_events provided -> history-based retrieval + re-ranking
      - If query provided -> query-based retrieval

    Falls back to V1 (content-based) if model artifacts are not available.
    """
    pipeline_version = getattr(request.app.state, "pipeline_version", "v1")
    pipeline = request.app.state.pipeline

    if pipeline_version == "v2":
        return _recommend_v2(pipeline, body)
    else:
        return _recommend_v1(pipeline, request, body)


def _recommend_v2(pipeline, body: RecommendationRequest) -> RecommendationResponse:
    """Full ML pipeline: Two-Tower retrieval -> LTR ranking -> MMR re-ranking."""
    if body.session_events and not body.user_id.startswith("user_"):
        # Anonymous user with session history
        item_ids = [e.item_id for e in body.session_events]
        raw_results = pipeline.recommend_by_history(item_ids, top_k=body.top_k)
        strategy = "history_based"
    elif body.user_id:
        raw_results = pipeline.recommend(body.user_id, top_k=body.top_k)
        strategy = "two_tower_ltr"
    else:
        raw_results = []
        strategy = "no_user"

    items = [
        RecommendedItem(
            item_id=r["item_id"],
            score=round(r["score"], 4),
            title=r.get("title", ""),
            category=r.get("category", ""),
            explanation=r.get("explanation", f"ML-ranked ({strategy})"),
        )
        for r in raw_results
    ]

    return RecommendationResponse(
        user_id=body.user_id,
        items=items,
        model_version="v2.0.0",
        explanation=f"Phase 2 ML pipeline -- {strategy}, {len(items)} items returned",
    )


def _recommend_v1(pipeline, request, body: RecommendationRequest) -> RecommendationResponse:
    """Phase 1 fallback: content-based retrieval."""
    embeddings = request.app.state.embeddings
    id_to_idx = request.app.state.id_to_idx

    if body.query:
        raw_results = pipeline.recommend_by_query(body.query, top_k=body.top_k)
        strategy = "query_based"
    elif body.session_events:
        item_ids = [e.item_id for e in body.session_events]
        raw_results = pipeline.recommend_by_history(
            item_ids, embeddings, id_to_idx, top_k=body.top_k
        )
        strategy = "history_based"
    else:
        raw_results = pipeline.recommend_popular(top_k=body.top_k)
        strategy = "popular_fallback"

    items = [
        RecommendedItem(
            item_id=r["item_id"],
            score=round(r["score"], 4),
            title=r.get("title", ""),
            category=r.get("category", ""),
            explanation=f"Content-based match (strategy={strategy})",
        )
        for r in raw_results
    ]

    return RecommendationResponse(
        user_id=body.user_id,
        items=items,
        model_version="mvp-0.1.0",
        explanation=f"Phase 1 MVP -- {strategy} retrieval, {len(items)} items returned",
    )
