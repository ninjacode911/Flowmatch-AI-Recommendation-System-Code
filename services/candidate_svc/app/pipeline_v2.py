"""
Recommendation Pipeline V2 (Phase 2 - Full ML Pipeline)

The complete multi-stage pipeline:
  1. RETRIEVAL: Two-Tower model generates candidate items via ANN search
  2. SCORING: LightGBM LTR ranks candidates using engineered features
  3. RE-RANKING: MMR diversity + freshness + business rules

This replaces the Phase 1 content-based pipeline with a proper
ML-powered recommendation system.

Pipeline flow:
  User request
    -> User Tower encodes user features (5ms)
    -> ANN search in Qdrant for top-500 candidates (10ms)
    -> Feature engineering for 500 (user, item) pairs (50ms)
    -> LightGBM scores and ranks candidates (5ms)
    -> Re-ranker applies diversity + business rules (2ms)
    -> Return top-K recommendations (~72ms total)
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch

from services.candidate_svc.app.vector_store import VectorStore
from services.reranking_svc.app.reranker import BusinessRules, Reranker
from services.training_pipeline.app.feature_engineering import FeatureEngineer
from services.training_pipeline.app.models.two_tower import TwoTowerModel


class RecommendationPipelineV2:
    """Phase 2 full ML pipeline: Two-Tower retrieval -> LTR ranking -> Re-ranking."""

    def __init__(
        self,
        vector_store: VectorStore,
        two_tower_model: TwoTowerModel,
        ltr_model: lgb.Booster,
        feature_eng: FeatureEngineer,
        reranker: Reranker,
        user_id_to_idx: dict[str, int],
        users: dict[str, dict],
        item_embeddings: np.ndarray,
    ) -> None:
        self.vector_store = vector_store
        self.two_tower = two_tower_model
        self.ltr = ltr_model
        self.feature_eng = feature_eng
        self.reranker = reranker
        self.user_id_to_idx = user_id_to_idx
        self.users = users
        self.item_embeddings = item_embeddings

    @torch.no_grad()
    def _encode_user(self, user_id: str) -> np.ndarray:
        """Encode user features with the User Tower."""
        user = self.users.get(user_id, {})
        user_idx = self.user_id_to_idx.get(user_id, 0)
        cluster_id = user.get("cluster_id", 0)
        age_norm = (user.get("age", 30) - 30) / 15.0
        gender = user.get("gender", "M")
        gender_onehot = [
            1.0 if gender == "M" else 0.0,
            1.0 if gender == "F" else 0.0,
            1.0 if gender == "NB" else 0.0,
        ]

        user_ids_t = torch.tensor([user_idx], dtype=torch.long)
        cluster_ids_t = torch.tensor([cluster_id], dtype=torch.long)
        user_features_t = torch.tensor([[age_norm] + gender_onehot], dtype=torch.float32)

        user_emb = self.two_tower.user_tower(user_ids_t, cluster_ids_t, user_features_t)
        return user_emb[0].cpu().numpy()

    def recommend(self, user_id: str, top_k: int = 10, num_candidates: int = 200) -> list[dict]:
        """
        Full pipeline: retrieve -> rank -> re-rank.

        Args:
            user_id: the user to recommend for
            top_k: number of final recommendations
            num_candidates: number of candidates from retrieval stage

        Returns:
            list of dicts with item_id, score, title, category, explanation
        """
        # Stage 1: RETRIEVAL -- Two-Tower user embedding -> ANN search
        user_vector = self._encode_user(user_id)
        candidates = self.vector_store.search(user_vector, top_k=num_candidates)

        if not candidates:
            return []

        # Stage 2: RANKING -- LightGBM LTR scores each candidate
        pairs = [(user_id, c["item_id"]) for c in candidates]
        features = self.feature_eng.compute_batch_features(pairs)
        ltr_scores = self.ltr.predict(features)

        # Merge LTR scores back into candidates
        for i, candidate in enumerate(candidates):
            candidate["score"] = float(ltr_scores[i])

        # Stage 3: RE-RANKING -- diversity + freshness + business rules
        # Get embeddings for the candidate items
        candidate_embs = []
        for c in candidates:
            idx = self.feature_eng.item_id_to_idx.get(c["item_id"])
            if idx is not None and self.item_embeddings is not None:
                candidate_embs.append(self.item_embeddings[idx])
            else:
                candidate_embs.append(np.zeros(256))
        candidate_emb_matrix = np.array(candidate_embs)

        reranked = self.reranker.rerank(candidates, candidate_emb_matrix, top_k=top_k)

        # Format output
        results = []
        for item in reranked:
            results.append({
                "item_id": item.item_id,
                "score": round(item.final_score, 4),
                "title": item.title,
                "category": item.category,
                "price": item.price,
                "explanation": item.explanation,
            })

        return results

    def recommend_by_history(
        self,
        item_ids: list[str],
        top_k: int = 10,
        num_candidates: int = 200,
    ) -> list[dict]:
        """
        Recommend based on item history (for anonymous users or cold-start).
        Falls back to averaging item embeddings like Phase 1.
        """
        valid_indices = [
            self.feature_eng.item_id_to_idx[iid]
            for iid in item_ids
            if iid in self.feature_eng.item_id_to_idx
        ]
        if not valid_indices or self.item_embeddings is None:
            return []

        user_embs = self.item_embeddings[valid_indices]
        user_vector = np.mean(user_embs, axis=0)
        user_vector = user_vector / (np.linalg.norm(user_vector) + 1e-8)

        candidates = self.vector_store.search(user_vector, top_k=num_candidates)

        # Re-rank with diversity (no LTR since we don't have user features)
        reranked = self.reranker.rerank(candidates, top_k=top_k)

        return [
            {
                "item_id": item.item_id,
                "score": round(item.final_score, 4),
                "title": item.title,
                "category": item.category,
                "price": item.price,
                "explanation": item.explanation,
            }
            for item in reranked
        ]


def build_pipeline_v2(data_dir: str | None = None) -> RecommendationPipelineV2:
    """
    Build the full Phase 2 pipeline from saved artifacts.

    Returns a ready-to-use RecommendationPipelineV2.
    """
    if data_dir is None:
        base = Path(__file__).resolve().parents[3]
    else:
        base = Path(data_dir).resolve().parents[1] if "synthetic" in data_dir else Path(data_dir).resolve().parent

    data_path = base / "data" / "synthetic"
    model_path = base / "models" / "artifacts"

    # Load items
    items = []
    with open(data_path / "items.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    print(f"Loaded {len(items):,} items")

    # Load users
    users = {}
    user_id_to_idx = {}
    with open(data_path / "users.jsonl") as f:
        for line in f:
            u = json.loads(line)
            user_id_to_idx[u["user_id"]] = len(user_id_to_idx)
            users[u["user_id"]] = u
    print(f"Loaded {len(users):,} users")

    # Load Two-Tower model
    two_tower = TwoTowerModel(
        num_users=len(user_id_to_idx),
        num_items=len(items),
        num_categories=8,
        num_clusters=20,
        output_dim=256,
        temperature=0.05,
    )
    checkpoint = torch.load(model_path / "two_tower_best.pt", weights_only=True)
    two_tower.load_state_dict(checkpoint["model_state_dict"])
    two_tower.eval()
    print(f"Loaded Two-Tower model (epoch {checkpoint['epoch']+1})")

    # Load Two-Tower item embeddings + index in Qdrant
    item_embeddings = np.load(model_path / "two_tower_item_embeddings.npy")
    print(f"Loaded item embeddings: {item_embeddings.shape}")

    store = VectorStore(in_memory=True)
    store.create_collection(vector_size=item_embeddings.shape[1])
    store.index_embeddings(item_embeddings, items)

    # Load LightGBM LTR
    ltr_model = lgb.Booster(model_file=str(model_path / "ltr_lightgbm.txt"))
    print("Loaded LightGBM LTR model")

    # Feature engineering
    feature_eng = FeatureEngineer(
        users_path=str(data_path / "users.jsonl"),
        items_path=str(data_path / "items.jsonl"),
        interactions_path=str(data_path / "interactions.jsonl"),
        item_embeddings_path=str(model_path / "two_tower_item_embeddings.npy"),
    )

    # Re-ranker
    reranker = Reranker(lambda_diversity=0.7, freshness_weight=0.05)

    pipeline = RecommendationPipelineV2(
        vector_store=store,
        two_tower_model=two_tower,
        ltr_model=ltr_model,
        feature_eng=feature_eng,
        reranker=reranker,
        user_id_to_idx=user_id_to_idx,
        users=users,
        item_embeddings=item_embeddings,
    )

    print("Pipeline V2 ready!")
    return pipeline
