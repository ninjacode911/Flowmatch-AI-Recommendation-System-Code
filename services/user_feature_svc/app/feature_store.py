"""
Feature Store -- serves pre-computed and real-time user/item features.

In a production recommendation system, features come from multiple sources:
  - Batch features: computed offline (user's purchase history, item popularity)
  - Real-time features: computed on the fly (clicks in the last 5 minutes)
  - Session features: computed per session (items viewed this visit)

This service stores features in Redis for low-latency access (<5ms).
Features are versioned so we can roll back if a bad feature is deployed.

Redis key design:
  user:{user_id}:features:v1  -> JSON blob of user features
  item:{item_id}:features:v1  -> JSON blob of item features
  session:{session_id}:items  -> list of item_ids viewed in this session
  user:{user_id}:recent       -> sorted set of recent interactions (score=timestamp)
"""

import json
import time

import structlog

logger = structlog.get_logger()

FEATURE_VERSION = "v1"
USER_FEATURE_TTL = 3600 * 24  # 24 hours
ITEM_FEATURE_TTL = 3600 * 6   # 6 hours
SESSION_TTL = 3600             # 1 hour
RECENT_INTERACTIONS_LIMIT = 100


class FeatureStore:
    """
    Redis-backed feature store for serving user and item features.

    Supports two modes:
      1. Redis mode: uses a real Redis connection (production)
      2. Local mode: uses an in-memory dict (development/testing)
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", local_mode: bool = True) -> None:
        self.local_mode = local_mode
        self._local_store: dict[str, str] = {}
        self._local_sorted_sets: dict[str, list[tuple[float, str]]] = {}
        self._local_lists: dict[str, list[str]] = {}
        self._redis = None

        if not local_mode:
            self._init_redis(redis_url)
        else:
            logger.info("feature_store.local_mode", msg="Using in-memory store")

    def _init_redis(self, redis_url: str) -> None:
        """Connect to Redis."""
        try:
            import redis

            self._redis = redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            logger.info("feature_store.redis_connected", url=redis_url)
        except ImportError:
            logger.warning("feature_store.redis_not_installed", msg="redis package not installed, using local mode")
            self.local_mode = True
        except Exception as e:
            logger.error("feature_store.redis_connection_failed", error=str(e))
            self.local_mode = True

    # ── User Features ─────────────────────────────────────────────────────

    def set_user_features(self, user_id: str, features: dict) -> None:
        """Store user features."""
        key = f"user:{user_id}:features:{FEATURE_VERSION}"
        value = json.dumps(features)

        if self.local_mode:
            self._local_store[key] = value
        else:
            self._redis.setex(key, USER_FEATURE_TTL, value)

    def get_user_features(self, user_id: str) -> dict | None:
        """Retrieve user features. Returns None if not found."""
        key = f"user:{user_id}:features:{FEATURE_VERSION}"

        if self.local_mode:
            raw = self._local_store.get(key)
        else:
            raw = self._redis.get(key)

        if raw is None:
            return None
        return json.loads(raw)

    # ── Item Features ─────────────────────────────────────────────────────

    def set_item_features(self, item_id: str, features: dict) -> None:
        """Store item features."""
        key = f"item:{item_id}:features:{FEATURE_VERSION}"
        value = json.dumps(features)

        if self.local_mode:
            self._local_store[key] = value
        else:
            self._redis.setex(key, ITEM_FEATURE_TTL, value)

    def get_item_features(self, item_id: str) -> dict | None:
        """Retrieve item features. Returns None if not found."""
        key = f"item:{item_id}:features:{FEATURE_VERSION}"

        if self.local_mode:
            raw = self._local_store.get(key)
        else:
            raw = self._redis.get(key)

        if raw is None:
            return None
        return json.loads(raw)

    def get_item_features_batch(self, item_ids: list[str]) -> dict[str, dict | None]:
        """Retrieve features for multiple items at once (batched for efficiency)."""
        if self.local_mode:
            return {iid: self.get_item_features(iid) for iid in item_ids}

        keys = [f"item:{iid}:features:{FEATURE_VERSION}" for iid in item_ids]
        values = self._redis.mget(keys)
        result = {}
        for iid, val in zip(item_ids, values):
            result[iid] = json.loads(val) if val else None
        return result

    # ── Session Tracking ──────────────────────────────────────────────────

    def add_session_item(self, session_id: str, item_id: str) -> None:
        """Track an item viewed in the current session."""
        key = f"session:{session_id}:items"

        if self.local_mode:
            if key not in self._local_lists:
                self._local_lists[key] = []
            if item_id not in self._local_lists[key]:
                self._local_lists[key].append(item_id)
        else:
            pipe = self._redis.pipeline()
            pipe.rpush(key, item_id)
            pipe.expire(key, SESSION_TTL)
            pipe.execute()

    def get_session_items(self, session_id: str) -> list[str]:
        """Get all items viewed in this session."""
        key = f"session:{session_id}:items"

        if self.local_mode:
            return self._local_lists.get(key, [])

        items = self._redis.lrange(key, 0, -1)
        return items or []

    # ── Recent Interactions ───────────────────────────────────────────────

    def add_recent_interaction(self, user_id: str, item_id: str, timestamp: float | None = None) -> None:
        """
        Track a recent user-item interaction.
        Uses a sorted set with timestamp as score for efficient time-range queries.
        """
        key = f"user:{user_id}:recent"
        ts = timestamp or time.time()

        if self.local_mode:
            if key not in self._local_sorted_sets:
                self._local_sorted_sets[key] = []
            self._local_sorted_sets[key].append((ts, item_id))
            # Keep only the most recent interactions
            self._local_sorted_sets[key].sort(key=lambda x: x[0], reverse=True)
            self._local_sorted_sets[key] = self._local_sorted_sets[key][:RECENT_INTERACTIONS_LIMIT]
        else:
            pipe = self._redis.pipeline()
            pipe.zadd(key, {item_id: ts})
            pipe.zremrangebyrank(key, 0, -RECENT_INTERACTIONS_LIMIT - 1)  # Trim to limit
            pipe.expire(key, USER_FEATURE_TTL)
            pipe.execute()

    def get_recent_interactions(self, user_id: str, limit: int = 20) -> list[str]:
        """Get the user's most recent interactions (newest first)."""
        key = f"user:{user_id}:recent"

        if self.local_mode:
            entries = self._local_sorted_sets.get(key, [])
            return [item_id for _, item_id in entries[:limit]]

        items = self._redis.zrevrange(key, 0, limit - 1)
        return items or []

    # ── Bulk Loading ──────────────────────────────────────────────────────

    def bulk_load_user_features(self, user_features: dict[str, dict]) -> int:
        """Load features for many users at once (used during batch feature computation)."""
        count = 0
        if self.local_mode:
            for user_id, features in user_features.items():
                self.set_user_features(user_id, features)
                count += 1
        else:
            pipe = self._redis.pipeline()
            for user_id, features in user_features.items():
                key = f"user:{user_id}:features:{FEATURE_VERSION}"
                pipe.setex(key, USER_FEATURE_TTL, json.dumps(features))
                count += 1
            pipe.execute()

        logger.info("feature_store.bulk_loaded_users", count=count)
        return count

    def bulk_load_item_features(self, item_features: dict[str, dict]) -> int:
        """Load features for many items at once."""
        count = 0
        if self.local_mode:
            for item_id, features in item_features.items():
                self.set_item_features(item_id, features)
                count += 1
        else:
            pipe = self._redis.pipeline()
            for item_id, features in item_features.items():
                key = f"item:{item_id}:features:{FEATURE_VERSION}"
                pipe.setex(key, ITEM_FEATURE_TTL, json.dumps(features))
                count += 1
            pipe.execute()

        logger.info("feature_store.bulk_loaded_items", count=count)
        return count

    # ── Stats ─────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return store statistics."""
        if self.local_mode:
            return {
                "mode": "local",
                "keys": len(self._local_store),
                "sorted_sets": len(self._local_sorted_sets),
                "lists": len(self._local_lists),
            }

        info = self._redis.info("keyspace")
        return {
            "mode": "redis",
            "feature_version": FEATURE_VERSION,
            "redis_info": info,
        }
