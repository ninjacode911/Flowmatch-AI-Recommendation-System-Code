"""
Evaluation Harness -- measures recommendation quality with standard IR metrics.

METRICS EXPLAINED:
  Recall@K: Of all items the user liked, how many did we find in top-K?
  NDCG@K: Did we rank the good items near the TOP of our list?
  Hit Rate@K: For what fraction of users did we find at least ONE relevant item?
  Coverage: What fraction of the catalogue did we recommend across all users?

Usage:
  python scripts/evaluate.py
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np



def load_interactions(path: str) -> dict[str, dict[str, list[str]]]:
    """Load interactions split by event type per user."""
    user_events: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            ix = json.loads(line)
            user_events[ix["user_id"]][ix["event_type"]].append(ix["item_id"])
    return dict(user_events)


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


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(base))
    data_dir = base / "data" / "synthetic"

    print("Loading data...")
    user_events = load_interactions(str(data_dir / "interactions.jsonl"))

    from services.candidate_svc.app.pipeline import build_pipeline_from_data
    pipeline, embeddings, id_to_idx = build_pipeline_from_data(str(data_dir))

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

    print(f"Evaluating {sample_size} users (out of {len(users_with_purchases)} with purchases)...")
    start = time.time()

    for uid in sampled_users:
        events = user_events[uid]
        view_history = events.get("view", []) + events.get("click", [])
        purchased = set(events.get("purchase", []))

        if not view_history or not purchased:
            continue

        results = pipeline.recommend_by_history(view_history[:20], embeddings, id_to_idx, top_k=max(k_values))
        rec_ids = [r["item_id"] for r in results]
        all_recommended_items.update(rec_ids)

        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(rec_ids, purchased, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(rec_ids, purchased, k))
            metrics[f"hit_rate@{k}"].append(hit_rate_at_k(rec_ids, purchased, k))

    elapsed = time.time() - start
    evaluated_users = len(sampled_users)

    print(f"\nEvaluation complete: {evaluated_users} users, {elapsed:.1f}s")
    print(f"Avg time per user: {elapsed/evaluated_users*1000:.1f}ms\n")
    print("=" * 50)
    print("PHASE 1 MVP -- OFFLINE EVALUATION RESULTS")
    print("=" * 50)
    print(f"{'Metric':<20} {'Value':>10} {'Target':>10}")
    print("-" * 50)

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
    print("\nNOTE: These are Phase 1 content-based baseline numbers.")
    print("Phase 2 (Two-Tower + LTR) will significantly improve these metrics.")


if __name__ == "__main__":
    main()
