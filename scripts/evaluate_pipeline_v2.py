"""
Full Pipeline V2 Evaluation -- end-to-end metrics for Phase 2.

Tests the complete pipeline:
  Two-Tower retrieval -> LightGBM LTR ranking -> MMR re-ranking

Compares against Phase 1 baseline (content-based retrieval only).

Usage:
  python scripts/evaluate_pipeline_v2.py
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base))


def load_interactions(path: str) -> dict[str, dict[str, list[str]]]:
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


def category_diversity(items: list[dict]) -> float:
    """Fraction of unique categories in the recommendation list."""
    if not items:
        return 0.0
    categories = set(item.get("category", "") for item in items)
    return len(categories) / len(items)


def main() -> None:
    data_dir = base / "data" / "synthetic"

    print("=" * 60)
    print("PHASE 2 FULL PIPELINE EVALUATION")
    print("=" * 60)

    print("\nLoading data...")
    user_events = load_interactions(str(data_dir / "interactions.jsonl"))

    print("Building Pipeline V2...")
    from services.candidate_svc.app.pipeline_v2 import build_pipeline_v2
    pipeline = build_pipeline_v2()

    k_values = [5, 10, 20]
    metrics: dict[str, list[float]] = {
        f"{m}@{k}": [] for k in k_values for m in ["recall", "ndcg", "hit_rate"]
    }
    metrics["diversity"] = []
    all_recommended_items: set[str] = set()

    users_with_purchases = [
        uid for uid, events in user_events.items()
        if events.get("purchase") and events.get("view")
    ]
    sample_size = min(200, len(users_with_purchases))
    rng = np.random.default_rng(42)
    sampled_users = list(rng.choice(users_with_purchases, size=sample_size, replace=False))

    print(f"\nEvaluating {sample_size} users with full pipeline (retrieval -> LTR -> re-rank)...")
    start = time.time()

    for uid in sampled_users:
        events = user_events[uid]
        purchased = set(events.get("purchase", []))

        if not purchased:
            continue

        results = pipeline.recommend(uid, top_k=max(k_values), num_candidates=200)
        rec_ids = [r["item_id"] for r in results]
        all_recommended_items.update(rec_ids)

        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(rec_ids, purchased, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(rec_ids, purchased, k))
            metrics[f"hit_rate@{k}"].append(hit_rate_at_k(rec_ids, purchased, k))

        metrics["diversity"].append(category_diversity(results))

    elapsed = time.time() - start
    evaluated = len(metrics["recall@5"])

    print(f"\nEvaluation complete: {evaluated} users, {elapsed:.1f}s")
    print(f"Avg time per user: {elapsed/max(evaluated,1)*1000:.1f}ms\n")

    print("=" * 60)
    print("PHASE 2 -- FULL PIPELINE EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<20} {'Value':>10} {'Target':>10}")
    print("-" * 60)

    targets = {
        "recall@10": 0.80, "recall@20": 0.80,
        "ndcg@10": 0.45, "ndcg@5": 0.45,
        "hit_rate@10": 0.70,
    }

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

    total_items = len(pipeline.feature_eng.items)
    coverage = len(all_recommended_items) / total_items * 100
    print(f"  {'coverage':<18} {coverage:>9.1f}% {'>=60%':>10}")

    print("\n" + "-" * 60)
    print("PIPELINE COMPARISON")
    print("-" * 60)
    print("  Phase 1 (content-based only):      ~0.0 on all metrics")
    print("  Phase 2 (Two-Tower only):           ~0.002 recall@20")
    recall_20 = np.mean(metrics["recall@20"]) if metrics["recall@20"] else 0
    ndcg_10 = np.mean(metrics["ndcg@10"]) if metrics["ndcg@10"] else 0
    hit_10 = np.mean(metrics["hit_rate@10"]) if metrics["hit_rate@10"] else 0
    diversity = np.mean(metrics["diversity"]) if metrics["diversity"] else 0
    print(f"  Phase 2 (full pipeline):            {recall_20:.4f} recall@20")
    print(f"                                      {ndcg_10:.4f} NDCG@10")
    print(f"                                      {hit_10:.4f} hit_rate@10")
    print(f"                                      {diversity:.4f} diversity")
    print()
    print("  The full pipeline should show improvement over retrieval-only")
    print("  because LTR re-ranks candidates using rich features, and the")
    print("  re-ranker adds category diversity.")

    # Show a sample recommendation
    sample_uid = sampled_users[0]
    print(f"\n{'='*60}")
    print(f"SAMPLE RECOMMENDATIONS for {sample_uid}")
    print(f"{'='*60}")
    sample_results = pipeline.recommend(sample_uid, top_k=10)
    for i, r in enumerate(sample_results):
        print(f"  {i+1:>2}. [{r['category']:<12}] {r['title'][:50]:<50} score={r['score']:.3f}  ${r['price']:.2f}")


if __name__ == "__main__":
    main()
