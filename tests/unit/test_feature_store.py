"""Unit tests for the FeatureStore (local mode)."""

from services.user_feature_svc.app.feature_store import FeatureStore


def test_feature_store_local_mode() -> None:
    store = FeatureStore(local_mode=True)
    assert store.local_mode is True
    assert store._redis is None


def test_user_features_roundtrip() -> None:
    store = FeatureStore(local_mode=True)
    features = {"age": 30, "cluster_id": 5, "total_purchases": 42}
    store.set_user_features("user_001", features)
    result = store.get_user_features("user_001")
    assert result == features


def test_user_features_not_found() -> None:
    store = FeatureStore(local_mode=True)
    assert store.get_user_features("nonexistent") is None


def test_item_features_roundtrip() -> None:
    store = FeatureStore(local_mode=True)
    features = {"category": "electronics", "price": 29.99, "rating": 4.5}
    store.set_item_features("item_100", features)
    result = store.get_item_features("item_100")
    assert result == features


def test_item_features_batch() -> None:
    store = FeatureStore(local_mode=True)
    store.set_item_features("item_A", {"category": "food"})
    store.set_item_features("item_B", {"category": "clothing"})

    batch = store.get_item_features_batch(["item_A", "item_B", "item_C"])
    assert batch["item_A"]["category"] == "food"
    assert batch["item_B"]["category"] == "clothing"
    assert batch["item_C"] is None


def test_session_tracking() -> None:
    store = FeatureStore(local_mode=True)
    store.add_session_item("sess_1", "item_A")
    store.add_session_item("sess_1", "item_B")
    store.add_session_item("sess_1", "item_C")
    items = store.get_session_items("sess_1")
    assert items == ["item_A", "item_B", "item_C"]


def test_session_dedup() -> None:
    store = FeatureStore(local_mode=True)
    store.add_session_item("sess_2", "item_A")
    store.add_session_item("sess_2", "item_A")  # duplicate
    items = store.get_session_items("sess_2")
    assert len(items) == 1


def test_empty_session() -> None:
    store = FeatureStore(local_mode=True)
    items = store.get_session_items("nonexistent_session")
    assert items == []


def test_recent_interactions() -> None:
    store = FeatureStore(local_mode=True)
    store.add_recent_interaction("user_1", "item_A", timestamp=100.0)
    store.add_recent_interaction("user_1", "item_B", timestamp=200.0)
    store.add_recent_interaction("user_1", "item_C", timestamp=300.0)

    recent = store.get_recent_interactions("user_1", limit=2)
    assert len(recent) == 2
    assert recent[0] == "item_C"  # most recent first
    assert recent[1] == "item_B"


def test_recent_interactions_empty() -> None:
    store = FeatureStore(local_mode=True)
    assert store.get_recent_interactions("no_user") == []


def test_bulk_load_users() -> None:
    store = FeatureStore(local_mode=True)
    users = {
        "u1": {"age": 25},
        "u2": {"age": 30},
        "u3": {"age": 35},
    }
    count = store.bulk_load_user_features(users)
    assert count == 3
    assert store.get_user_features("u1")["age"] == 25
    assert store.get_user_features("u3")["age"] == 35


def test_bulk_load_items() -> None:
    store = FeatureStore(local_mode=True)
    items = {
        "i1": {"price": 10.0},
        "i2": {"price": 20.0},
    }
    count = store.bulk_load_item_features(items)
    assert count == 2


def test_stats() -> None:
    store = FeatureStore(local_mode=True)
    store.set_user_features("u1", {"x": 1})
    store.set_item_features("i1", {"y": 2})
    stats = store.get_stats()
    assert stats["mode"] == "local"
    assert stats["keys"] == 2
