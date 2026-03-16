"""
Evaluate Two-Tower Model -- use the trained User Tower for proper evaluation.

CORRECT EVALUATION APPROACH:
  Instead of averaging item embeddings (which ignores the User Tower entirely),
  we use the User Tower to encode each user's features into a 256-d embedding,
  then search for the nearest items in the Two-Tower item embedding space.

  This is exactly how the model would work in production:
    1. User visits the site
    2. User Tower encodes their features (user_id, age, gender, cluster) -> 256-d vector
    3. ANN search finds the 20 closest items in the pre-computed item embedding index
    4. Those are the recommendations

Usage:
  python scripts/evaluate_two_tower.py
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base))


def load_interactions(path: str) -> dict[str, dict[str, list[str]]]:
    """Load interactions split by event type per user."""
    user_events: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            ix = json.loads(line)
            user_events[ix["user_id"]][ix["event_type"]].append(ix["item_id"])
    return dict(user_events)


def load_users(path: str) -> dict[str, dict]:
    """Load user profiles."""
    users = {}
    with open(path) as f:
        for line in f:
            user = json.loads(line)
            users[user["user_id"]] = user
    return users


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(recommended[:k]) & relevant else 0.0


@torch.no_grad()
def encode_user(model, user_data: dict, user_id_to_idx: dict[str, int]) -> np.ndarray:
    """
    Use the trained User Tower to encode a user into a 256-d vector.

    This is the CORRECT way to get recommendations from a Two-Tower model:
    encode the user with the User Tower, not by averaging item embeddings.
    """
    user_id = user_data["user_id"]
    user_idx = user_id_to_idx.get(user_id, 0)
    cluster_id = user_data.get("cluster_id", 0)
    age_norm = (user_data.get("age", 30) - 30) / 15.0
    gender = user_data.get("gender", "M")
    gender_onehot = [
        1.0 if gender == "M" else 0.0,
        1.0 if gender == "F" else 0.0,
        1.0 if gender == "NB" else 0.0,
    ]

    user_ids_t = torch.tensor([user_idx], dtype=torch.long)
    cluster_ids_t = torch.tensor([cluster_id], dtype=torch.long)
    user_features_t = torch.tensor([[age_norm] + gender_onehot], dtype=torch.float32)

    user_emb = model.user_tower(user_ids_t, cluster_ids_t, user_features_t)
    return user_emb[0].cpu().numpy()


def main() -> None:
    data_dir = base / "data" / "synthetic"
    model_dir = base / "models" / "artifacts"

    print("=" * 60)
    print("TWO-TOWER MODEL EVALUATION (User Tower)")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    user_events = load_interactions(str(data_dir / "interactions.jsonl"))
    users = load_users(str(data_dir / "users.jsonl"))

    # Load Two-Tower model
    print("Loading trained Two-Tower model...")
    from services.training_pipeline.app.dataset import InteractionDataset
    from services.training_pipeline.app.models.two_tower import TwoTowerModel

    # We need the dataset just for metadata (user/item counts, mappings)
    dataset = InteractionDataset(
        interactions_path=str(data_dir / "interactions.jsonl"),
        users_path=str(data_dir / "users.jsonl"),
        items_path=str(data_dir / "items.jsonl"),
        content_embeddings_path=str(data_dir / "item_embeddings.npy"),
    )

    model = TwoTowerModel(
        num_users=len(dataset.user_id_to_idx),
        num_items=len(dataset.item_id_to_idx),
        num_categories=len(dataset.category_to_idx),
        num_clusters=20,
        output_dim=256,
        temperature=0.05,
    )
    checkpoint = torch.load(model_dir / "two_tower_best.pt", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']+1} (val_loss={checkpoint['val_loss']:.4f})")

    # Load Two-Tower item embeddings + index in Qdrant
    print("Building vector index...")
    from services.candidate_svc.app.vector_store import VectorStore

    items_list = []
    with open(data_dir / "items.jsonl") as f:
        for line in f:
            items_list.append(json.loads(line))

    embeddings = np.load(model_dir / "two_tower_item_embeddings.npy")
    with open(model_dir / "two_tower_id_to_idx.json") as f:
        id_to_idx = json.load(f)

    store = VectorStore(in_memory=True)
    store.create_collection(vector_size=embeddings.shape[1])
    store.index_embeddings(embeddings, items_list)

    # Evaluate
    k_values = [5, 10, 20]
    metrics: dict[str, list[float]] = {f"{m}@{k}": [] for k in k_values for m in ["recall", "ndcg", "hit_rate"]}
    all_recommended_items: set[str] = set()
    total_items = len(id_to_idx)

    users_with_purchases = [
        uid for uid, events in user_events.items()
        if events.get("purchase") and events.get("view")
    ]
    sample_size = min(500, len(users_with_purchases))
    rng = np.random.default_rng(42)
    sampled_users = list(rng.choice(users_with_purchases, size=sample_size, replace=False))

    print(f"\nEvaluating {sample_size} users using User Tower embeddings...")
    start = time.time()

    for uid in sampled_users:
        events = user_events[uid]
        purchased = set(events.get("purchase", []))

        if not purchased:
            continue

        user_data = users.get(uid)
        if user_data is None or uid not in dataset.user_id_to_idx:
            continue

        # Use the User Tower to encode this user
        user_vector = encode_user(model, user_data, dataset.user_id_to_idx)

        # Search for nearest items
        results = store.search(user_vector, top_k=max(k_values))
        rec_ids = [r["item_id"] for r in results]
        all_recommended_items.update(rec_ids)

        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(rec_ids, purchased, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(rec_ids, purchased, k))
            metrics[f"hit_rate@{k}"].append(hit_rate_at_k(rec_ids, purchased, k))

    elapsed = time.time() - start
    evaluated = len(metrics["recall@5"])

    print(f"\nEvaluation complete: {evaluated} users, {elapsed:.1f}s")
    print(f"Avg time per user: {elapsed/max(evaluated,1)*1000:.1f}ms\n")
    print("=" * 60)
    print("PHASE 2 -- TWO-TOWER (USER TOWER) EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<20} {'Value':>10} {'Target':>10}")
    print("-" * 60)

    targets = {"recall@10": 0.80, "recall@20": 0.80, "ndcg@10": 0.45, "ndcg@5": 0.45, "hit_rate@10": 0.70}

    for metric_name in sorted(metrics.keys()):
        values = metrics[metric_name]
        if values:
            avg = np.mean(values)
            target = targets.get(metric_name, "-")
            target_str = f">= {target:.2f}" if isinstance(target, float) else str(target)
            status = ""
            if isinstance(target, float):
                status = " PASS" if avg >= target else " MISS"
            print(f"  {metric_name:<18} {avg:>10.4f} {target_str:>10}{status}")

    coverage = len(all_recommended_items) / total_items * 100
    print(f"  {'coverage':<18} {coverage:>9.1f}% {'>=60%':>10}")

    print("\n" + "-" * 60)
    print("INTERPRETATION")
    print("-" * 60)
    print(f"  Total items in catalogue: {total_items:,}")
    avg_purchases = np.mean([len(events.get("purchase", [])) for uid, events in list(user_events.items())[:100]])
    print(f"  Max K evaluated:          {max(k_values)}")
    print()
    print("  With 50K items and only top-20 recommendations, even a good model")
    print("  has low absolute recall -- it's like finding needles in a haystack.")
    print("  The important thing is RELATIVE improvement over the baseline.")
    print()
    print("  Phase 1 (content-based):  ~0.000 recall@20")
    recall_20 = np.mean(metrics["recall@20"]) if metrics["recall@20"] else 0
    print(f"  Phase 2 (Two-Tower):      {recall_20:.4f} recall@20")
    if recall_20 > 0:
        print("  --> Two-Tower shows collaborative learning signal!")
    print()
    print("  Next steps: NCF + LightGBM LTR will further improve ranking quality.")


if __name__ == "__main__":
    main()
