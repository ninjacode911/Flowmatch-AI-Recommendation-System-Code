"""
Vector Store — manages the Qdrant index for ANN (Approximate Nearest Neighbour) search.

HOW ANN SEARCH WORKS:
  Traditional search: compare query against EVERY item -> O(N) = too slow for millions of items.
  ANN search: build an index (HNSW graph) that lets you find the ~top-500 most similar items
  in O(log N) time -- milliseconds instead of minutes.

  HNSW (Hierarchical Navigable Small World) works like a skip list for vectors:
    - Bottom layer: all items connected to nearby neighbours
    - Upper layers: sparse connections for long-range jumps
    - Search: start at top layer, greedily move toward query, drop down, repeat
    - Result: finds approximate (not exact) nearest neighbours, but ~95%+ recall

WHAT THIS MODULE DOES:
  1. create_collection() -- creates a Qdrant collection with HNSW config
  2. index_embeddings() -- uploads item vectors + metadata (payload) to Qdrant
  3. search() -- given a query vector, find the top-K most similar items
"""

import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from shared.config.settings import settings


class VectorStore:
    """Manages the Qdrant vector index for item embeddings."""

    def __init__(
        self,
        host: str = settings.qdrant_host,
        port: int = settings.qdrant_port,
        collection_name: str = settings.qdrant_collection,
        in_memory: bool = False,
    ) -> None:
        """
        Connect to Qdrant.

        in_memory=True uses Qdrant's embedded mode (no Docker needed for dev).
        in_memory=False connects to a running Qdrant server.
        """
        if in_memory:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    def create_collection(self, vector_size: int = settings.embedding_dim) -> None:
        """
        Create or recreate the Qdrant collection.

        HNSW Parameters explained:
          - m=16: each node connects to 16 neighbours (higher = better recall, more memory)
          - ef_construct=200: beam width during index build (higher = better index, slower build)

        We use COSINE distance because our embeddings are L2-normalized.
        """
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name in collections:
            print(f"Collection '{self.collection_name}' already exists. Recreating...")
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
                on_disk=False,
            ),
        )
        print(f"Created collection '{self.collection_name}' (dim={vector_size}, distance=cosine)")

    def index_embeddings(
        self,
        embeddings: np.ndarray,
        items: list[dict],
        batch_size: int = 500,
    ) -> None:
        """Upload item embeddings + metadata to Qdrant in batches."""
        total = len(items)
        assert embeddings.shape[0] == total, f"Mismatch: {embeddings.shape[0]} embeddings vs {total} items"

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            points = []

            for idx in range(start, end):
                item = items[idx]
                points.append(
                    PointStruct(
                        id=idx,
                        vector=embeddings[idx].tolist(),
                        payload={
                            "item_id": item["item_id"],
                            "title": item["title"],
                            "category": item["category"],
                            "subcategory": item.get("subcategory", ""),
                            "price": item.get("price", 0.0),
                            "rating": item.get("rating", 0.0),
                            "brand": item.get("brand", ""),
                            "popularity_score": item.get("popularity_score", 0.0),
                        },
                    )
                )

            self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"  Indexed {end:,} / {total:,} items")

        print(f"Indexing complete. Collection '{self.collection_name}' has {total:,} vectors.")

    def search(
        self,
        query_vector: list[float] | np.ndarray,
        top_k: int = 10,
        category_filter: str | None = None,
    ) -> list[dict]:
        """Find the top-K most similar items to the query vector."""
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        query_filter = None
        if category_filter:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_filter = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=category_filter))]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )

        return [
            {
                "item_id": hit.payload["item_id"],
                "score": hit.score,
                "title": hit.payload.get("title", ""),
                "category": hit.payload.get("category", ""),
                "price": hit.payload.get("price", 0.0),
                "rating": hit.payload.get("rating", 0.0),
                "brand": hit.payload.get("brand", ""),
            }
            for hit in results.points
        ]

    def collection_info(self) -> dict:
        """Get collection stats."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": str(info.status),
        }


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[3]
    items_path = str(base / "data" / "synthetic" / "items.jsonl")
    embeddings_path = str(base / "data" / "synthetic" / "item_embeddings.npy")

    items = []
    with open(items_path) as f:
        for line in f:
            items.append(json.loads(line))
    print(f"Loaded {len(items):,} items")

    embeddings = np.load(embeddings_path)
    print(f"Loaded embeddings: {embeddings.shape}")

    store = VectorStore(in_memory=True)
    store.create_collection(vector_size=embeddings.shape[1])
    store.index_embeddings(embeddings, items)
    print(f"\nCollection info: {store.collection_info()}")
