"""
LightGBM Learning-to-Rank (LTR) Training Script.

=== WHAT IS LEARNING-TO-RANK? ===

LTR is the final ranking stage in the recommendation pipeline.
Given a set of candidate items (from Two-Tower retrieval), LTR re-orders them
by predicted relevance using rich features.

=== WHY LIGHTGBM? ===

LightGBM with LambdaRank is the industry standard for LTR because:
  1. Fast training (minutes, not hours like neural rankers)
  2. Handles mixed feature types naturally (categorical + numerical)
  3. Directly optimizes NDCG (the metric we care about)
  4. Easy to inspect feature importance (which features matter most?)
  5. Robust to missing features and outliers

=== LAMBDARANK OBJECTIVE ===

Instead of pointwise (predict relevance score) or pairwise (predict which is better),
LambdaRank is a LISTWISE objective that directly optimizes NDCG@K.

It works by computing gradients that push relevant items higher in the ranked list,
weighted by how much swapping two items would change NDCG. Items near the top of
the list get larger gradients (because NDCG cares more about top positions).

=== DATA FORMAT ===

LTR data is organized in "queries" (groups):
  - Each query = one user
  - Each document = one candidate item
  - Label = relevance grade (0=no interaction, 1=view, 2=click, 3=purchase)
  - Features = engineered features from FeatureEngineer

Usage:
  python -m services.training_pipeline.app.train_ltr
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from services.training_pipeline.app.feature_engineering import FeatureEngineer

# ── Hyperparameters ──────────────────────────────────────────────────────────

NUM_CANDIDATES = 100      # Number of candidate items per user for training (was 50)
NUM_TRAIN_USERS = 5000    # Users for training (was 2000)
NUM_VAL_USERS = 1500      # Users for validation (was 500)
RELEVANCE_GRADES = {"purchase": 3, "add_to_cart": 2, "click": 1, "view": 0}

DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"


def build_ltr_dataset(
    user_events: dict[str, dict[str, list[str]]],
    feature_eng: FeatureEngineer,
    user_ids: list[str],
    all_item_ids: list[str],
    num_candidates: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build LTR training data in the format LightGBM expects.

    For each user (query):
      - Include items they interacted with (positive, with relevance grades)
      - Sample random items they didn't interact with (negative, grade=0)
      - Compute features for each (user, item) pair

    Returns:
      features: (total_pairs, num_features)
      labels: (total_pairs,) relevance grades
      groups: list of group sizes (one per user)

    WHY GROUPS?
      LambdaRank needs to know which items belong to the same user (query).
      It optimizes the ranking WITHIN each group, not across groups.
    """
    all_features = []
    all_labels = []
    groups = []
    rng = np.random.default_rng(42)

    # Pre-convert item IDs to numpy array for fast sampling
    all_item_ids_arr = np.array(all_item_ids)
    total_users = len(user_ids)

    for user_num, uid in enumerate(user_ids):
        if (user_num + 1) % 500 == 0:
            print(f"      {user_num+1:,} / {total_users:,} users processed...")

        events = user_events.get(uid, {})
        if not events:
            continue

        # Build relevance map for this user's items
        item_relevance: dict[str, int] = {}
        for event_type, grade in RELEVANCE_GRADES.items():
            for iid in events.get(event_type, []):
                # Keep highest grade per item
                if iid not in item_relevance or grade > item_relevance[iid]:
                    item_relevance[iid] = grade

        # Positive items (user interacted with)
        positive_items = list(item_relevance.keys())

        # Sample negative items (vectorized)
        positive_set = set(positive_items)
        num_negatives = max(0, num_candidates - len(positive_items))
        if num_negatives > 0:
            # Sample more than needed, then filter
            candidates = rng.choice(all_item_ids_arr, size=num_negatives * 3, replace=False)
            negative_items = [c for c in candidates if c not in positive_set][:num_negatives]
        else:
            negative_items = []

        # Combine and compute features
        candidate_items = positive_items + negative_items
        if not candidate_items:
            continue

        group_features = []
        group_labels = []

        for iid in candidate_items:
            feats = feature_eng.compute_features(uid, iid)
            label = item_relevance.get(iid, 0)
            group_features.append(feats)
            group_labels.append(label)

        all_features.extend(group_features)
        all_labels.extend(group_labels)
        groups.append(len(candidate_items))

    return (
        np.array(all_features, dtype=np.float32),
        np.array(all_labels, dtype=np.float32),
        np.array(groups, dtype=np.int32),
    )


def main() -> None:
    print("=" * 60)
    print("LIGHTGBM LEARNING-TO-RANK TRAINING")
    print("=" * 60)

    # ── 1. Load interaction data ─────────────────────────────────────────
    print("\n[1/5] Loading interaction data...")
    user_events: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with open(DATA_DIR / "interactions.jsonl") as f:
        for line in f:
            ix = json.loads(line)
            user_events[ix["user_id"]][ix["event_type"]].append(ix["item_id"])
    user_events = dict(user_events)

    all_item_ids = []
    with open(DATA_DIR / "items.jsonl") as f:
        for line in f:
            item = json.loads(line)
            all_item_ids.append(item["item_id"])
    print(f"  {len(user_events):,} users, {len(all_item_ids):,} items")

    # ── 2. Initialize feature engineering ────────────────────────────────
    print("\n[2/5] Computing features...")
    tt_emb_path = str(MODEL_DIR / "two_tower_item_embeddings.npy")
    feature_eng = FeatureEngineer(
        users_path=str(DATA_DIR / "users.jsonl"),
        items_path=str(DATA_DIR / "items.jsonl"),
        interactions_path=str(DATA_DIR / "interactions.jsonl"),
        item_embeddings_path=tt_emb_path,
    )
    feature_names = feature_eng.get_feature_names()
    print(f"  {len(feature_names)} features: {feature_names[:5]}...")

    # ── 3. Build train/val datasets ──────────────────────────────────────
    print("\n[3/5] Building LTR datasets...")
    # Select users who have purchases (meaningful training signal)
    users_with_purchases = [
        uid for uid, events in user_events.items()
        if events.get("purchase")
    ]
    rng = np.random.default_rng(42)
    rng.shuffle(users_with_purchases)

    train_users = users_with_purchases[:NUM_TRAIN_USERS]
    val_users = users_with_purchases[NUM_TRAIN_USERS:NUM_TRAIN_USERS + NUM_VAL_USERS]

    print(f"  Building train set ({len(train_users)} users)...")
    start = time.time()
    train_features, train_labels, train_groups = build_ltr_dataset(
        user_events, feature_eng, train_users, all_item_ids, NUM_CANDIDATES
    )
    print(f"    Train: {train_features.shape[0]:,} pairs, {len(train_groups)} groups ({time.time()-start:.1f}s)")

    print(f"  Building val set ({len(val_users)} users)...")
    start = time.time()
    val_features, val_labels, val_groups = build_ltr_dataset(
        user_events, feature_eng, val_users, all_item_ids, NUM_CANDIDATES
    )
    print(f"    Val: {val_features.shape[0]:,} pairs, {len(val_groups)} groups ({time.time()-start:.1f}s)")

    # Print label distribution
    for split_name, labels in [("Train", train_labels), ("Val", val_labels)]:
        unique, counts = np.unique(labels, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        print(f"    {split_name} label distribution: {dist}")

    # ── 4. Train LightGBM ───────────────────────────────────────────────
    print("\n[4/5] Training LightGBM LambdaRank...")
    train_data = lgb.Dataset(
        train_features,
        label=train_labels,
        group=train_groups,
        feature_name=feature_names,
    )
    val_data = lgb.Dataset(
        val_features,
        label=val_labels,
        group=val_groups,
        feature_name=feature_names,
        reference=train_data,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5, 10, 20],
        "num_leaves": 63,            # More complex trees for larger dataset (was 31)
        "learning_rate": 0.03,       # Slightly higher LR with more data (was 0.02)
        "min_data_in_leaf": 30,      # Lower for finer splits (was 50)
        "feature_fraction": 0.8,     # Use more features per tree (was 0.7)
        "bagging_fraction": 0.8,     # Use more data per tree (was 0.7)
        "bagging_freq": 1,
        "max_depth": 8,              # Deeper trees for richer patterns (was 6)
        "lambda_l1": 0.05,           # Slightly less L1 reg (was 0.1)
        "lambda_l2": 0.5,            # Slightly less L2 reg (was 1.0)
        "verbose": 1,
        "num_threads": -1,           # Use all CPU cores (was 4)
        "seed": 42,
        "lambdarank_truncation_level": 20,
    }

    print(f"  Parameters: {json.dumps({k: v for k, v in params.items() if k != 'verbose'}, indent=2)}")

    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=100),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # ── 5. Save model and report ─────────────────────────────────────────
    print("\n[5/5] Saving model and feature importance...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "ltr_lightgbm.txt"
    model.save_model(str(model_path))
    print(f"  Saved model: {model_path}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feature_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 60)
    print("LTR TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n  Best iteration: {model.best_iteration}")
    print(f"  Best NDCG@10 (val): {model.best_score['val']['ndcg@10']:.4f}")

    print("\n  Top 15 features by importance (gain):")
    for fname, imp in feature_imp[:15]:
        bar = "#" * int(imp / max(importance) * 30)
        print(f"    {fname:<30} {imp:>10.1f}  {bar}")

    print(f"\n  Model: {model_path}")
    print("  Next: Build re-ranker and integrate all models.")


if __name__ == "__main__":
    main()
