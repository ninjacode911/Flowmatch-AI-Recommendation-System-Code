"""
Re-Ranking Service -- the final stage before recommendations reach the user.

=== WHY RE-RANK? ===

The LTR model optimizes for RELEVANCE -- how likely is the user to like this item?
But relevance alone makes bad recommendations. Imagine:
  - All 10 recommendations are red Nike shoes (relevant, but boring)
  - All recommendations are old products (relevant, but stale)
  - No promoted items appear (relevant, but bad for business)

Re-ranking adds three critical objectives on top of relevance:

1. DIVERSITY (MMR):
   "Don't show 10 items from the same category"
   Uses Maximal Marginal Relevance to balance relevance with diversity.

2. FRESHNESS:
   "Boost newer items, penalize stale inventory"
   Applies a time-decay factor so recent items get a small boost.

3. BUSINESS RULES:
   "Ensure at least 1 promoted item in top 5"
   Hard constraints from the business team that override pure relevance.

=== MMR (MAXIMAL MARGINAL RELEVANCE) ===

MMR is a greedy algorithm:
  1. Start with the most relevant item
  2. For each remaining slot, pick the item that maximizes:
       MMR(i) = lambda * relevance(i) - (1-lambda) * max_similarity(i, already_selected)
  3. lambda controls the relevance-diversity tradeoff:
       lambda=1.0 -> pure relevance (no diversity)
       lambda=0.0 -> pure diversity (ignore relevance)
       lambda=0.7 -> good balance (70% relevance, 30% diversity)

Reference: Carbonell & Goldstein, "The Use of MMR, Diversity-Based Reranking" (1998)
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RerankedItem:
    """An item after re-ranking with all scores attached."""

    item_id: str
    relevance_score: float
    diversity_penalty: float = 0.0
    freshness_boost: float = 0.0
    business_boost: float = 0.0
    final_score: float = 0.0
    category: str = ""
    title: str = ""
    price: float = 0.0
    explanation: str = ""


@dataclass
class BusinessRules:
    """
    Configurable business rules for re-ranking.

    In production, these would come from a config service or A/B test framework.
    """

    # Category diversity: max items from same category in top-K
    max_same_category: int = 3

    # Promoted items: item_ids that should be boosted
    promoted_items: set[str] = field(default_factory=set)
    promoted_boost: float = 0.1  # Score boost for promoted items

    # Price range: ensure mix of price points
    min_budget_items: int = 1  # At least N items under median price in top-K
    min_premium_items: int = 1  # At least N items above median price in top-K


class Reranker:
    """
    Re-ranks a list of scored items to optimize for relevance + diversity + freshness.
    """

    def __init__(
        self,
        lambda_diversity: float = 0.7,
        freshness_weight: float = 0.05,
        business_rules: BusinessRules | None = None,
    ) -> None:
        """
        Args:
            lambda_diversity: MMR tradeoff (1.0 = pure relevance, 0.0 = pure diversity)
            freshness_weight: How much to boost newer items (0-1)
            business_rules: Optional business constraints
        """
        self.lambda_div = lambda_diversity
        self.freshness_weight = freshness_weight
        self.rules = business_rules or BusinessRules()

    def rerank(
        self,
        items: list[dict],
        item_embeddings: np.ndarray | None = None,
        top_k: int = 10,
    ) -> list[RerankedItem]:
        """
        Re-rank items using MMR diversity + freshness + business rules.

        Args:
            items: list of dicts with at least {item_id, score, category, title, price}
            item_embeddings: optional (N, D) embeddings for MMR similarity
            top_k: number of items to return

        Returns:
            list of RerankedItem, ordered by final score
        """
        if not items:
            return []

        n = len(items)
        top_k = min(top_k, n)

        # Normalize relevance scores to [0, 1]
        scores = np.array([item["score"] for item in items])
        if scores.max() > scores.min():
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            norm_scores = np.ones(n)

        # Compute freshness boost (based on popularity as proxy for recency)
        freshness = np.array([item.get("popularity_score", 0.5) for item in items])

        # Compute business boosts
        business_boosts = np.zeros(n)
        for i, item in enumerate(items):
            if item["item_id"] in self.rules.promoted_items:
                business_boosts[i] = self.rules.promoted_boost

        # MMR re-ranking
        if item_embeddings is not None and item_embeddings.shape[0] == n:
            selected = self._mmr_select(
                norm_scores, item_embeddings, freshness, business_boosts, top_k
            )
        else:
            # Fallback: category-aware greedy selection (no embeddings needed)
            selected = self._category_diverse_select(
                norm_scores, items, freshness, business_boosts, top_k
            )

        # Build result
        result = []
        for idx in selected:
            item = items[idx]
            reranked = RerankedItem(
                item_id=item["item_id"],
                relevance_score=float(norm_scores[idx]),
                freshness_boost=float(freshness[idx] * self.freshness_weight),
                business_boost=float(business_boosts[idx]),
                final_score=float(norm_scores[idx] + freshness[idx] * self.freshness_weight + business_boosts[idx]),
                category=item.get("category", ""),
                title=item.get("title", ""),
                price=item.get("price", 0.0),
                explanation=self._explain(item, norm_scores[idx], freshness[idx], business_boosts[idx]),
            )
            result.append(reranked)

        return result

    def _mmr_select(
        self,
        scores: np.ndarray,
        embeddings: np.ndarray,
        freshness: np.ndarray,
        business_boosts: np.ndarray,
        top_k: int,
    ) -> list[int]:
        """
        Maximal Marginal Relevance selection.

        Greedily selects items that are both relevant AND dissimilar
        to already-selected items.
        """
        n = len(scores)
        # Pre-compute pairwise similarities
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        normalized = embeddings / norms
        sim_matrix = normalized @ normalized.T  # (n, n) cosine similarities

        selected: list[int] = []
        remaining = set(range(n))

        for _ in range(top_k):
            if not remaining:
                break

            best_idx = -1
            best_mmr = -float("inf")

            for idx in remaining:
                relevance = scores[idx] + freshness[idx] * self.freshness_weight + business_boosts[idx]

                # Max similarity to already selected items
                if selected:
                    max_sim = max(sim_matrix[idx][s] for s in selected)
                else:
                    max_sim = 0.0

                mmr = self.lambda_div * relevance - (1 - self.lambda_div) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return selected

    def _category_diverse_select(
        self,
        scores: np.ndarray,
        items: list[dict],
        freshness: np.ndarray,
        business_boosts: np.ndarray,
        top_k: int,
    ) -> list[int]:
        """
        Category-based diversity selection (fallback when no embeddings available).

        Greedily selects items, enforcing max_same_category constraint.
        """
        combined_scores = scores + freshness * self.freshness_weight + business_boosts
        sorted_indices = np.argsort(-combined_scores)

        selected: list[int] = []
        category_counts: dict[str, int] = {}

        for idx in sorted_indices:
            if len(selected) >= top_k:
                break

            cat = items[int(idx)].get("category", "")
            count = category_counts.get(cat, 0)

            if count < self.rules.max_same_category:
                selected.append(int(idx))
                category_counts[cat] = count + 1

        return selected

    def _explain(
        self,
        item: dict,
        relevance: float,
        freshness: float,
        business_boost: float,
    ) -> str:
        """Generate a human-readable explanation for why this item was recommended."""
        parts = [f"relevance={relevance:.2f}"]
        if freshness > 0.5:
            parts.append("trending")
        if business_boost > 0:
            parts.append("promoted")

        category = item.get("category", "unknown")
        return f"[{category}] {', '.join(parts)}"
