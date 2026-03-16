"""
End-to-end tests simulating the full service pipeline flow.

Tests the interaction between services:
  Event Collector -> User Feature Store -> LLM Augmentation

These tests use local/mock mode for all external dependencies.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from services.event_collector.app.main import app as event_app
from services.event_collector.app.producer import EventProducer
from services.llm_augment_svc.app.main import LLMClient
from services.llm_augment_svc.app.main import app as llm_app
from services.user_feature_svc.app.feature_store import FeatureStore
from services.user_feature_svc.app.main import app as feature_app


@pytest.fixture(autouse=True)
def _setup_all_app_states():
    """Initialize app state for all services."""
    event_app.state.producer = EventProducer(local_mode=True)
    feature_app.state.store = FeatureStore(local_mode=True)
    llm_app.state.llm = LLMClient(local_mode=True)


@pytest.fixture
async def event_client():
    transport = ASGITransport(app=event_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def feature_client():
    transport = ASGITransport(app=feature_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def llm_client():
    transport = ASGITransport(app=llm_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_full_user_journey(
    event_client: AsyncClient,
    feature_client: AsyncClient,
    llm_client: AsyncClient,
) -> None:
    """
    Simulate a complete user journey through the recommendation system:
    1. User browses items (events captured)
    2. Session items are tracked in the feature store
    3. Recommendations are generated and explained by the LLM service
    """
    user_id = "e2e_user_001"
    session_id = "e2e_session_001"

    # Step 1: User views several items (events sent to event collector)
    viewed_items = [
        {"user_id": user_id, "item_id": "item_A", "event_type": "view", "session_id": session_id},
        {"user_id": user_id, "item_id": "item_B", "event_type": "view", "session_id": session_id},
        {"user_id": user_id, "item_id": "item_C", "event_type": "click", "session_id": session_id},
    ]

    batch_resp = await event_client.post("/events/batch", json={"events": viewed_items})
    assert batch_resp.status_code == 200
    assert batch_resp.json()["events_received"] == 3

    # Step 2: Track session items in feature store
    for item in viewed_items:
        resp = await feature_client.post(
            f"/features/session/{session_id}",
            json={"item_id": item["item_id"]},
        )
        assert resp.status_code == 200

    # Verify session contains all items
    session_resp = await feature_client.get(f"/features/session/{session_id}")
    assert session_resp.status_code == 200
    session_data = session_resp.json()
    assert session_data["count"] == 3

    # Step 3: Store user features
    store = feature_app.state.store
    store.set_user_features(user_id, {"age": 28, "cluster_id": 2, "preferred_categories": ["electronics"]})
    store.add_recent_interaction(user_id, "item_C", timestamp=1000.0)

    user_resp = await feature_client.get(f"/features/user/{user_id}")
    assert user_resp.status_code == 200
    assert user_resp.json()["features"]["age"] == 28

    # Step 4: Get explanations for recommended items
    explain_payload = {
        "user_id": user_id,
        "items": [
            {"item_id": "item_X", "title": "Wireless Mouse", "category": "electronics", "score": 0.92, "price": 25.99},
            {"item_id": "item_Y", "title": "USB Hub", "category": "electronics", "score": 0.78, "price": 18.50},
        ],
        "user_context": "Interested in electronics accessories",
    }
    explain_resp = await llm_client.post("/explain", json=explain_payload)
    assert explain_resp.status_code == 200
    explain_data = explain_resp.json()
    assert len(explain_data["explanations"]) == 2
    assert explain_data["explanations"][0]["item_id"] == "item_X"

    # Step 5: User searches (query understanding)
    query_resp = await llm_client.post("/parse-query", json={
        "query": "recommend cheap wireless accessories",
        "user_id": user_id,
    })
    assert query_resp.status_code == 200
    parsed = query_resp.json()["parsed"]
    assert parsed["intent"] == "recommendation"
    assert parsed["price_range"] == "budget"
    assert "wireless" in parsed["attributes"]

    # Step 6: User purchases (event captured)
    purchase_resp = await event_client.post("/events", json={
        "user_id": user_id,
        "item_id": "item_X",
        "event_type": "purchase",
        "price": 25.99,
        "source": "recommendations",
        "session_id": session_id,
    })
    assert purchase_resp.status_code == 200

    # Verify event stats reflect activity
    stats_resp = await event_client.get("/events/stats")
    stats = stats_resp.json()
    assert stats["total_events_processed"] >= 4


@pytest.mark.asyncio
async def test_cold_start_user(
    feature_client: AsyncClient,
    llm_client: AsyncClient,
) -> None:
    """Test handling of a brand-new user with no history."""
    new_user = "e2e_new_user"

    # No features exist for this user
    resp = await feature_client.get(f"/features/user/{new_user}")
    assert resp.status_code == 404

    # No recent interactions
    recent_resp = await feature_client.get(f"/features/recent/{new_user}")
    assert recent_resp.status_code == 200
    assert recent_resp.json()["count"] == 0

    # LLM service still produces explanations (template mode)
    explain_resp = await llm_client.post("/explain", json={
        "user_id": new_user,
        "items": [
            {"item_id": "pop_1", "title": "Best Seller", "category": "books", "score": 0.5, "price": 14.99},
        ],
    })
    assert explain_resp.status_code == 200
    assert len(explain_resp.json()["explanations"]) == 1
