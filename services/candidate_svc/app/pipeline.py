"""
Recommendation Pipeline (Phase 1 - MVP)

Orchestrates:
  1. Embedding Service -> encode user query into a vector
  2. Candidate Service -> ANN search in Qdrant for top-K similar items
  3. (Phase 2: Ranking -> LTR scoring)
  4. (Phase 2: Re-ranking -> diversity, freshness)

For Phase 1, the pipeline is simple:
  - If the user provides a text query -> embed it -> ANN search
  - If the user has session history -> average the embeddings of viewed items -> ANN search
  - Cold start (no query, no history) -> return popular items
"""

import json
from pathlib import Path

import numpy as np

from services.embedding_svc.app.embedder import ItemEmbedder
from services.candidate_svc.app.vector_store import VectorStore


class RecommendationPipeline:
    """Phase 1 MVP pipeline: content-based retrieval via embeddings + ANN search."""

    def __init__(self, vector_store: VectorStore, embedder: ItemEmbedder) -> None:
        self.vector_store = vector_store
        self.embedder = embedder

    def recommend_by_query(self, query: str, top_k: int = 10) -> list[dict]:
        """Recommend items based on a natural language query."""
        query_vector = self.embedder.embed_text([query])[0]
        return self.vector_store.search(query_vector, top_k=top_k)

    def recommend_by_history(
        self,
        item_ids: list[str],
        embeddings_matrix: np.ndarray,
        id_to_idx: dict[str, int],
        top_k: int = 10,
    ) -> list[dict]:
        """
        Recommend items based on user's interaction history.

        Averages the embeddings of items the user interacted with to create
        a "user taste vector", then searches for similar items.
        """
        valid_indices = [id_to_idx[iid] for iid in item_ids if iid in id_to_idx]
        if not valid_indices:
            return []

        user_embs = embeddings_matrix[valid_indices]
        user_vector = np.mean(user_embs, axis=0)
        user_vector = user_vector / (np.linalg.norm(user_vector) + 1e-8)

        return self.vector_store.search(user_vector, top_k=top_k)

    def recommend_popular(self, top_k: int = 10) -> list[dict]:
        """Fallback: return popular items (cold-start users)."""
        seed_query = "popular trending bestseller product"
        return self.recommend_by_query(seed_query, top_k=top_k)


def build_pipeline_from_data(data_dir: str | None = None) -> tuple[RecommendationPipeline, np.ndarray, dict[str, int]]:
    """
    Build the full MVP pipeline from pre-computed data files.

    Returns: (pipeline, embeddings_matrix, id_to_idx_map)
    """
    if data_dir is None:
        data_dir = str(Path(__file__).resolve().parents[3] / "data" / "synthetic")

    data_path = Path(data_dir)

    items = []
    with open(data_path / "items.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    print(f"Loaded {len(items):,} items")

    embeddings = np.load(data_path / "item_embeddings.npy")
    print(f"Loaded embeddings: {embeddings.shape}")

    id_to_idx = {item["item_id"]: idx for idx, item in enumerate(items)}

    store = VectorStore(in_memory=True)
    store.create_collection(vector_size=embeddings.shape[1])
    store.index_embeddings(embeddings, items)

    embedder = ItemEmbedder()

    pipeline = RecommendationPipeline(vector_store=store, embedder=embedder)
    return pipeline, embeddings, id_to_idx


if __name__ == "__main__":
    pipeline, embeddings, id_to_idx = build_pipeline_from_data()

    print("\n" + "=" * 60)
    print("QUERY-BASED RECOMMENDATIONS")
    print("=" * 60)
    for query in ["wireless bluetooth headphones", "organic skincare products", "beginner yoga equipment"]:
        print(f"\nQuery: '{query}'")
        results = pipeline.recommend_by_query(query, top_k=5)
        for i, r in enumerate(results):
            print(f"  {i+1}. [{r['category']}] {r['title']} (score={r['score']:.3f}, ${r['price']:.2f})")

    print("\n" + "=" * 60)
    print("HISTORY-BASED RECOMMENDATIONS")
    print("=" * 60)
    history_items = ["item_000000", "item_000008", "item_000016"]
    print(f"\nUser history: {history_items}")
    results = pipeline.recommend_by_history(history_items, embeddings, id_to_idx, top_k=5)
    for i, r in enumerate(results):
        print(f"  {i+1}. [{r['category']}] {r['title']} (score={r['score']:.3f}, ${r['price']:.2f})")
