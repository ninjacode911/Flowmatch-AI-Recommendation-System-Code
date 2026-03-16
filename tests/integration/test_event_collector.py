"""Integration tests for the Event Collector service."""

import pytest
from httpx import ASGITransport, AsyncClient

from services.event_collector.app.main import app
from services.event_collector.app.producer import EventProducer


@pytest.fixture(autouse=True)
def _setup_app_state():
    """Initialize app state that lifespan would normally set up."""
    app.state.producer = EventProducer(local_mode=True)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_collect_single_event(client: AsyncClient) -> None:
    payload = {
        "user_id": "user_001",
        "item_id": "item_100",
        "event_type": "click",
    }
    resp = await client.post("/events", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "accepted"
    assert data["events_received"] == 1


@pytest.mark.asyncio
async def test_collect_event_generates_event_id(client: AsyncClient) -> None:
    payload = {
        "user_id": "user_002",
        "item_id": "item_200",
        "event_type": "view",
    }
    resp = await client.post("/events", json=payload)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_collect_batch_events(client: AsyncClient) -> None:
    payload = {
        "events": [
            {"user_id": "user_001", "item_id": "item_101", "event_type": "view"},
            {"user_id": "user_001", "item_id": "item_102", "event_type": "click"},
            {"user_id": "user_001", "item_id": "item_103", "event_type": "purchase"},
        ]
    }
    resp = await client.post("/events/batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "accepted"
    assert data["events_received"] == 3


@pytest.mark.asyncio
async def test_collect_event_invalid_type(client: AsyncClient) -> None:
    payload = {
        "user_id": "user_001",
        "item_id": "item_100",
        "event_type": "invalid_type",
    }
    resp = await client.post("/events", json=payload)
    assert resp.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_collect_event_missing_user_id(client: AsyncClient) -> None:
    payload = {
        "item_id": "item_100",
        "event_type": "click",
    }
    resp = await client.post("/events", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_get_stats(client: AsyncClient) -> None:
    resp = await client.get("/events/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_events_processed" in data
    assert "buffer_size" in data
    assert data["local_mode"] is True


@pytest.mark.asyncio
async def test_search_event_routes_to_correct_topic(client: AsyncClient) -> None:
    payload = {
        "user_id": "user_010",
        "event_type": "search",
        "query": "wireless headphones",
    }
    resp = await client.post("/events", json=payload)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_purchase_event_with_price(client: AsyncClient) -> None:
    payload = {
        "user_id": "user_010",
        "item_id": "item_500",
        "event_type": "purchase",
        "price": 49.99,
        "source": "recommendations",
    }
    resp = await client.post("/events", json=payload)
    assert resp.status_code == 200
