"""
Item Embedding Service — converts item text into dense vectors.

HOW IT WORKS:
  1. Takes an item's title + description as input text
  2. Passes it through a pre-trained SentenceTransformer model (all-MiniLM-L6-v2)
  3. Returns a 384-dimensional vector that captures the item's semantic meaning

WHY MiniLM?
  - It's small (80MB) and fast — encodes text in ~5ms per item on CPU
  - 384 dimensions is a good balance between quality and storage cost
  - It's the most popular model for this task in production systems
  - In Phase 2, we'll also add CLIP for image embeddings and fuse them together

THE KEY INSIGHT:
  Items with similar descriptions will have similar embeddings (close in vector space).
  So when a user looks at "wireless noise-cancelling headphones", the system can find
  "Bluetooth ANC earbuds" even though the words are completely different — because
  the *meaning* is similar.
"""

import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Auto-detect GPU (RTX 5070 works in WSL2 with PyTorch 2.7+)
import torch as _torch

DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"

# Model is downloaded once and cached in ~/.cache/torch/sentence_transformers/
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class ItemEmbedder:
    """Encodes item text into dense vector embeddings."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """
        Load the pre-trained model.
        First call downloads ~80MB model; subsequent calls use cache.
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, device=DEVICE)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dim}")

    def embed_text(self, texts: list[str]) -> np.ndarray:
        """
        Encode a batch of texts into embeddings.

        Args:
            texts: List of strings (e.g., ["Blue running shoes from Nike", ...])

        Returns:
            numpy array of shape (len(texts), 384)
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 100,
            batch_size=256,
            normalize_embeddings=True,  # L2 normalize so cosine sim = dot product
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_items(self, items: list[dict]) -> list[dict]:
        """
        Embed a list of item dicts. Each item must have 'title' and 'description'.

        Returns list of dicts with item_id and embedding vector.
        """
        # Combine title + description for richer text representation
        texts = [f"{item['title']}. {item['description']}" for item in items]
        embeddings = self.embed_text(texts)

        results = []
        for item, emb in zip(items, embeddings):
            results.append({
                "item_id": item["item_id"],
                "embedding": emb.tolist(),
            })
        return results


def embed_catalogue(items_path: str, output_path: str) -> None:
    """
    Batch-embed the entire item catalogue and save to disk.

    This is run offline (not in the request path). The embeddings are then
    loaded into Qdrant for fast ANN search.
    """
    # Load items
    items = []
    with open(items_path) as f:
        for line in f:
            items.append(json.loads(line))
    print(f"Loaded {len(items):,} items from {items_path}")

    # Embed in batches
    embedder = ItemEmbedder()
    batch_size = 1000
    all_embeddings = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        texts = [f"{item['title']}. {item['description']}" for item in batch]
        embs = embedder.embed_text(texts)
        all_embeddings.append(embs)
        print(f"  Embedded {min(i + batch_size, len(items)):,} / {len(items):,}")

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Final embedding matrix: {embeddings.shape}")

    # Save as .npy for fast loading
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, embeddings)
    print(f"Saved embeddings to {output}")

    # Also save item_id → index mapping
    id_map = {item["item_id"]: idx for idx, item in enumerate(items)}
    map_path = output.with_suffix(".json")
    with open(map_path, "w") as f:
        json.dump(id_map, f)
    print(f"Saved ID map to {map_path}")


if __name__ == "__main__":
    import sys

    base = Path(__file__).resolve().parents[3]
    items_path = str(base / "data" / "synthetic" / "items.jsonl")
    output_path = str(base / "data" / "synthetic" / "item_embeddings.npy")

    if "--embed" in sys.argv:
        embed_catalogue(items_path, output_path)
    else:
        # Quick demo: embed a few sample texts
        embedder = ItemEmbedder()
        samples = [
            "Premium wireless noise-cancelling headphones",
            "Bluetooth ANC earbuds with deep bass",
            "Organic cotton yoga pants for women",
        ]
        embs = embedder.embed_text(samples)
        print(f"\nEmbedding shape: {embs.shape}")

        # Show similarity between items
        from numpy.linalg import norm

        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                sim = float(np.dot(embs[i], embs[j]) / (norm(embs[i]) * norm(embs[j])))
                print(f"  Similarity({samples[i][:40]}..., {samples[j][:40]}...): {sim:.3f}")
