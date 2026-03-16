"""Unit tests for shared Pydantic schemas."""

from shared.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendedItem,
)


def test_recommendation_request_defaults() -> None:
    """A request with no args should use sensible defaults."""
    req = RecommendationRequest()
    assert req.user_id == ""
    assert req.session_events == []
    assert req.query is None
    assert req.top_k == 10


def test_recommendation_request_with_user() -> None:
    req = RecommendationRequest(user_id="user_123", top_k=20)
    assert req.user_id == "user_123"
    assert req.top_k == 20


def test_recommendation_response_empty() -> None:
    resp = RecommendationResponse(user_id="user_1", items=[], model_version="v0.1")
    assert resp.user_id == "user_1"
    assert len(resp.items) == 0


def test_recommended_item() -> None:
    item = RecommendedItem(
        item_id="item_42",
        score=0.95,
        title="Blue Jacket",
        category="clothing",
        explanation="Similar to items you viewed recently",
    )
    assert item.item_id == "item_42"
    assert item.score == 0.95
