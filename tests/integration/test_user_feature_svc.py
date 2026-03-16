"""Integration tests for the User Feature Service."""

import pytest
from httpx import ASGITransport, AsyncClient

from services.user_feature_svc.app.feature_store import FeatureStore
from services.user_feature_svc.app.main import app


@pytest.fixture(autouse=True)
def _setup_app_state():
    """Initialize app state that lifespan would normally set up."""
    app.state.store = FeatureStore(local_mode=True)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def seeded_client(client: AsyncClient):
    """Client with some features pre-loaded via the store."""
    store = app.state.store
    store.set_user_features("user_001", {"age": 25, "cluster_id": 3, "total_clicks": 150})
    store.set_item_features("item_100", {"category": "electronics", "price": 29.99, "rating": 4.5})
    store.set_item_features("item_101", {"category": "clothing", "price": 59.99, "rating": 4.0})
    return client


@pytest.mark.asyncio
async def test_get_user_features(seeded_client: AsyncClient) -> None:
    resp = await seeded_client.get("/features/user/user_001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "user_001"
    assert data["features"]["age"] == 25
    assert data["features"]["cluster_id"] == 3


@pytest.mark.asyncio
async def test_get_user_features_not_found(client: AsyncClient) -> None:
    resp = await client.get("/features/user/nonexistent_user")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_item_features(seeded_client: AsyncClient) -> None:
    resp = await seeded_client.get("/features/item/item_100")
    assert resp.status_code == 200
    data = resp.json()
    assert data["item_id"] == "item_100"
    assert data["features"]["category"] == "electronics"


@pytest.mark.asyncio
async def test_get_item_features_not_found(client: AsyncClient) -> None:
    resp = await client.get("/features/item/nonexistent_item")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_item_features_batch(seeded_client: AsyncClient) -> None:
    resp = await seeded_client.post("/features/items", json={"item_ids": ["item_100", "item_101", "item_999"]})
    assert resp.status_code == 200
    data = resp.json()
    items = data["items"]
    assert items["item_100"]["category"] == "electronics"
    assert items["item_101"]["category"] == "clothing"
    assert items["item_999"] is None


@pytest.mark.asyncio
async def test_session_tracking(client: AsyncClient) -> None:
    sid = "session_abc123"

    # Add items to session
    await client.post(f"/features/session/{sid}", json={"item_id": "item_001"})
    await client.post(f"/features/session/{sid}", json={"item_id": "item_002"})
    await client.post(f"/features/session/{sid}", json={"item_id": "item_003"})

    # Retrieve session
    resp = await client.get(f"/features/session/{sid}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == sid
    assert data["count"] == 3
    assert "item_001" in data["items"]
    assert "item_003" in data["items"]


@pytest.mark.asyncio
async def test_recent_interactions(client: AsyncClient) -> None:
    store = app.state.store
    store.add_recent_interaction("user_050", "item_A", timestamp=1000.0)
    store.add_recent_interaction("user_050", "item_B", timestamp=2000.0)
    store.add_recent_interaction("user_050", "item_C", timestamp=3000.0)

    resp = await client.get("/features/recent/user_050?limit=2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    # Most recent first
    assert data["recent_items"][0] == "item_C"
    assert data["recent_items"][1] == "item_B"


@pytest.mark.asyncio
async def test_feature_store_stats(client: AsyncClient) -> None:
    resp = await client.get("/features/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "local"
