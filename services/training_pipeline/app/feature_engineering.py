"""
Feature Engineering for Learning-to-Rank (LTR).

=== WHAT IS FEATURE ENGINEERING? ===

The LTR model (LightGBM) doesn't work with raw data directly. Instead, we
compute "features" -- numerical signals that help the model predict relevance.

Features fall into 5 categories:

1. USER FEATURES: who is this person?
   - age, gender, cluster membership
   - how active they are (total interactions)
   - category preferences (what fraction of their clicks are in each category)

2. ITEM FEATURES: what is this product?
   - price, rating, popularity score
   - category (one-hot encoded)
   - how many total interactions it has received

3. CROSS FEATURES: how does this user relate to this item?
   - Does the item's category match the user's preferred category?
   - Price relative to user's average spend
   - Two-Tower similarity score (dot product of user/item embeddings)
   - NCF prediction score

4. CONTEXT FEATURES: what's happening right now?
   - Time of day, day of week (would be used in production)
   - Session depth (how many items viewed so far)

5. EMBEDDING FEATURES: compressed neural network signals
   - Two-Tower user embedding (256-d -> top principal components)
   - Two-Tower item embedding similarity to user history

=== WHY SO MANY FEATURES? ===

LightGBM is a gradient-boosted decision tree. Unlike neural networks,
it can't learn complex transformations from raw inputs. But it's VERY
good at combining many hand-crafted features with non-linear splits.

Think of it like this:
  - Neural nets: learn features automatically, need lots of data
  - GBDTs: need engineered features, but work great with less data

In production recommender systems, the LTR stage typically has 50-200 features.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


class FeatureEngineer:
    """Computes features for (user, item) pairs for the LTR model."""

    def __init__(
        self,
        users_path: str,
        items_path: str,
        interactions_path: str,
        item_embeddings_path: str | None = None,
    ) -> None:
        # Load users
        self.users: dict[str, dict] = {}
        with open(users_path) as f:
            for line in f:
                u = json.loads(line)
                self.users[u["user_id"]] = u

        # Load items
        self.items: dict[str, dict] = {}
        self.item_id_to_idx: dict[str, int] = {}
        with open(items_path) as f:
            for line in f:
                item = json.loads(line)
                self.item_id_to_idx[item["item_id"]] = len(self.item_id_to_idx)
                self.items[item["item_id"]] = item

        # Load item embeddings (Two-Tower, if available)
        self.item_embeddings: np.ndarray | None = None
        if item_embeddings_path and Path(item_embeddings_path).exists():
            self.item_embeddings = np.load(item_embeddings_path)

        # Compute user interaction stats
        self.user_interaction_count: dict[str, int] = Counter()
        self.user_category_dist: dict[str, dict[str, float]] = {}
        self.user_avg_price: dict[str, float] = {}
        self.item_interaction_count: dict[str, int] = Counter()

        self._compute_interaction_stats(interactions_path)

        # Compute global stats for normalization
        prices = [item["price"] for item in self.items.values()]
        self.price_mean = np.mean(prices)
        self.price_std = np.std(prices) + 1e-8
        ratings = [item.get("rating", 3.0) for item in self.items.values()]
        self.rating_mean = np.mean(ratings)
        self.rating_std = np.std(ratings) + 1e-8

        # All categories for one-hot encoding
        self.categories = sorted(set(item["category"] for item in self.items.values()))
        self.cat_to_idx = {c: i for i, c in enumerate(self.categories)}

    def _compute_interaction_stats(self, interactions_path: str) -> None:
        """Compute per-user and per-item statistics from interaction data."""
        user_cats: dict[str, list[str]] = defaultdict(list)
        user_prices: dict[str, list[float]] = defaultdict(list)

        with open(interactions_path) as f:
            for line in f:
                ix = json.loads(line)
                uid = ix["user_id"]
                iid = ix["item_id"]

                self.user_interaction_count[uid] += 1
                self.item_interaction_count[iid] += 1

                if iid in self.items:
                    item = self.items[iid]
                    user_cats[uid].append(item["category"])
                    user_prices[uid].append(item["price"])

        # Compute category distributions per user
        for uid, cats in user_cats.items():
            total = len(cats)
            cat_counts = Counter(cats)
            self.user_category_dist[uid] = {c: cnt / total for c, cnt in cat_counts.items()}

        # Compute average price per user
        for uid, prices in user_prices.items():
            self.user_avg_price[uid] = np.mean(prices)

    def get_feature_names(self) -> list[str]:
        """Return ordered list of feature names."""
        names = [
            # User features
            "user_age_norm",
            "user_gender_M",
            "user_gender_F",
            "user_gender_NB",
            "user_cluster_id",
            "user_interaction_count",
        ]
        # User category preferences
        for cat in self.categories:
            names.append(f"user_pref_{cat}")

        names += [
            "user_avg_price_norm",
            # Item features
            "item_price_norm",
            "item_rating_norm",
            "item_popularity",
            "item_interaction_count",
        ]
        # Item category one-hot
        for cat in self.categories:
            names.append(f"item_cat_{cat}")

        names += [
            # Cross features
            "cross_cat_match",         # Does item category match user's top category?
            "cross_cat_preference",    # User's preference score for this item's category
            "cross_price_ratio",       # Item price / user average price
            "cross_price_diff_norm",   # Normalized price difference
        ]

        # Embedding similarity (if available)
        if self.item_embeddings is not None:
            names.append("cross_emb_popularity_bucket")

        return names

    def compute_features(self, user_id: str, item_id: str) -> np.ndarray:
        """
        Compute feature vector for a single (user, item) pair.

        Returns a 1-D numpy array of floats.
        """
        user = self.users.get(user_id, {})
        item = self.items.get(item_id, {})

        features = []

        # ── User features ────────────────────────────────────────────
        age = user.get("age", 30)
        features.append((age - 30) / 15.0)  # age normalized

        gender = user.get("gender", "M")
        features.append(1.0 if gender == "M" else 0.0)
        features.append(1.0 if gender == "F" else 0.0)
        features.append(1.0 if gender == "NB" else 0.0)

        features.append(float(user.get("cluster_id", 0)))

        ix_count = self.user_interaction_count.get(user_id, 0)
        features.append(np.log1p(ix_count))  # log-scale interaction count

        # User category preferences
        user_cat_dist = self.user_category_dist.get(user_id, {})
        for cat in self.categories:
            features.append(user_cat_dist.get(cat, 0.0))

        avg_price = self.user_avg_price.get(user_id, self.price_mean)
        features.append((avg_price - self.price_mean) / self.price_std)

        # ── Item features ────────────────────────────────────────────
        price = item.get("price", 0)
        features.append((price - self.price_mean) / self.price_std)

        rating = item.get("rating", 3.0)
        features.append((rating - self.rating_mean) / self.rating_std)

        features.append(item.get("popularity_score", 0.0))

        item_ix_count = self.item_interaction_count.get(item_id, 0)
        features.append(np.log1p(item_ix_count))

        # Item category one-hot
        item_cat = item.get("category", "")
        for cat in self.categories:
            features.append(1.0 if cat == item_cat else 0.0)

        # ── Cross features ───────────────────────────────────────────
        # Category match: does the item's category match user's top category?
        top_user_cat = max(user_cat_dist, key=user_cat_dist.get) if user_cat_dist else ""
        features.append(1.0 if item_cat == top_user_cat else 0.0)

        # User's preference for this item's category
        features.append(user_cat_dist.get(item_cat, 0.0))

        # Price ratio: item price relative to user's typical spend
        if avg_price > 0:
            features.append(price / avg_price)
        else:
            features.append(1.0)

        # Normalized price difference
        features.append(abs(price - avg_price) / self.price_std)

        # Embedding-based feature
        if self.item_embeddings is not None:
            idx = self.item_id_to_idx.get(item_id)
            if idx is not None:
                # Use embedding norm as a proxy for "how distinctive" this item is
                emb_norm = np.linalg.norm(self.item_embeddings[idx])
                features.append(emb_norm)
            else:
                features.append(0.0)

        return np.array(features, dtype=np.float32)

    def compute_batch_features(
        self,
        pairs: list[tuple[str, str]],
    ) -> np.ndarray:
        """Compute features for a batch of (user_id, item_id) pairs."""
        return np.array([self.compute_features(uid, iid) for uid, iid in pairs])
