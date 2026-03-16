"""Integration tests for the LLM Augmentation Service."""

import pytest
from httpx import ASGITransport, AsyncClient

from services.llm_augment_svc.app.main import LLMClient, app


@pytest.fixture(autouse=True)
def _setup_app_state():
    """Initialize app state that lifespan would normally set up."""
    app.state.llm = LLMClient(local_mode=True)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_explain_recommendations(client: AsyncClient) -> None:
    payload = {
        "user_id": "user_001",
        "items": [
            {"item_id": "item_1", "title": "Wireless Headphones", "category": "electronics", "score": 0.95, "price": 79.99},
            {"item_id": "item_2", "title": "Running Shoes", "category": "sports", "score": 0.82, "price": 120.00},
            {"item_id": "item_3", "title": "Organic Tea", "category": "food", "score": 0.65, "price": 12.99},
        ],
    }
    resp = await client.post("/explain", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "user_001"
    assert len(data["explanations"]) == 3
    assert data["model_used"] == "template"
    # First item gets "Top pick" explanation
    assert "Top pick" in data["explanations"][0]["explanation"]


@pytest.mark.asyncio
async def test_explain_with_user_context(client: AsyncClient) -> None:
    payload = {
        "user_id": "user_002",
        "items": [
            {"item_id": "item_10", "title": "Yoga Mat", "category": "sports", "score": 0.9, "price": 35.0},
        ],
        "user_context": "Interested in fitness",
    }
    resp = await client.post("/explain", json=payload)
    assert resp.status_code == 200
    assert len(resp.json()["explanations"]) == 1


@pytest.mark.asyncio
async def test_parse_search_query(client: AsyncClient) -> None:
    payload = {"query": "affordable wireless headphones", "user_id": "user_001"}
    resp = await client.post("/parse-query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    parsed = data["parsed"]
    assert parsed["intent"] == "search"
    assert parsed["price_range"] == "budget"  # "affordable" triggers budget
    assert "wireless" in parsed["attributes"]
    assert data["model_used"] == "template"


@pytest.mark.asyncio
async def test_parse_recommendation_query(client: AsyncClient) -> None:
    payload = {"query": "recommend me the best electronics"}
    resp = await client.post("/parse-query", json=payload)
    assert resp.status_code == 200
    parsed = resp.json()["parsed"]
    assert parsed["intent"] == "recommendation"
    assert "electronics" in parsed["categories"]


@pytest.mark.asyncio
async def test_parse_comparison_query(client: AsyncClient) -> None:
    payload = {"query": "compare these two sports products"}
    resp = await client.post("/parse-query", json=payload)
    assert resp.status_code == 200
    parsed = resp.json()["parsed"]
    assert parsed["intent"] == "comparison"
    assert "sports" in parsed["categories"]


@pytest.mark.asyncio
async def test_parse_question_query(client: AsyncClient) -> None:
    payload = {"query": "how do premium headphones work?"}
    resp = await client.post("/parse-query", json=payload)
    assert resp.status_code == 200
    parsed = resp.json()["parsed"]
    assert parsed["intent"] == "question"
    assert parsed["price_range"] == "premium"


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["local_mode"] is True
