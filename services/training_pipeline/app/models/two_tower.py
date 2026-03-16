"""
Two-Tower Retrieval Model — the industry-standard for large-scale candidate retrieval.

=== WHAT IS A TWO-TOWER MODEL? ===

Imagine you're Netflix. You have 200M users and 50K movies. To recommend movies,
you COULD score every (user, movie) pair — but that's 10 TRILLION comparisons.
Impossible in real-time.

The Two-Tower trick:
  1. Build a "User Tower" — a neural network that takes user features and outputs
     a 256-dimensional vector (the "user embedding").
  2. Build an "Item Tower" — a neural network that takes item features and outputs
     a 256-dimensional vector (the "item embedding").
  3. At serving time:
     - Run the User Tower ONCE to get the user's embedding (~5ms)
     - Use ANN search to find the 500 closest item embeddings (~10ms)
     - Done! No need to score every item.

The item embeddings are pre-computed and stored in Qdrant.
Only the user tower runs at inference time — that's why it's fast.

=== HOW IS IT TRAINED? ===

We train it on (user, item) pairs from interaction data:
  - Positive pair: user clicked/purchased this item
  - Negative pair: random items the user didn't interact with

The training objective: make positive pairs have HIGH dot-product similarity,
and negative pairs have LOW similarity.

This is called "in-batch negatives" or "sampled softmax":
  - In a batch of 256 (user, item) pairs
  - Each user's positive item is treated as the target
  - All OTHER items in the batch are treated as negatives
  - The loss pushes the correct pair's similarity above all negatives

=== ARCHITECTURE ===

User Tower:
  Input: [user_id_embedding(64-d) | user_features(age, gender, etc.)]
  → Linear(128) → ReLU → BatchNorm → Linear(256) → L2-normalize

Item Tower:
  Input: [item_id_embedding(64-d) | content_embedding(384-d) | item_features]
  → Linear(256) → ReLU → BatchNorm → Linear(256) → L2-normalize

Output: dot_product(user_emb, item_emb) = relevance score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """
    Encodes user features into a dense embedding.

    Input features:
      - user_id: integer index → learned embedding
      - age: normalized float
      - gender: one-hot encoded (3 categories: M, F, NB)
      - cluster_id: integer → learned embedding (captures user segment)
    """

    def __init__(
        self,
        num_users: int,
        num_clusters: int = 20,
        user_emb_dim: int = 64,
        cluster_emb_dim: int = 16,
        num_extra_features: int = 4,  # age + gender (3 one-hot)
        hidden_dim: int = 128,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        self.cluster_embedding = nn.Embedding(num_clusters, cluster_emb_dim)

        input_dim = user_emb_dim + cluster_emb_dim + num_extra_features
        self.tower = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        cluster_ids: torch.Tensor,
        user_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            user_ids: (batch_size,) integer user indices
            cluster_ids: (batch_size,) integer cluster indices
            user_features: (batch_size, 4) float tensor [age, gender_M, gender_F, gender_NB]

        Returns:
            (batch_size, 256) L2-normalized user embeddings
        """
        u_emb = self.user_embedding(user_ids)
        c_emb = self.cluster_embedding(cluster_ids)
        x = torch.cat([u_emb, c_emb, user_features], dim=-1)
        out = self.tower(x)
        return F.normalize(out, p=2, dim=-1)


class ItemTower(nn.Module):
    """
    Encodes item features into a dense embedding.

    Input features:
      - item_id: integer index → learned embedding
      - content_embedding: 384-d from SentenceTransformer (pre-computed)
      - category: integer → learned embedding
      - price, rating, popularity: normalized floats
    """

    def __init__(
        self,
        num_items: int,
        num_categories: int = 8,
        item_emb_dim: int = 64,
        cat_emb_dim: int = 16,
        content_emb_dim: int = 384,
        num_extra_features: int = 3,  # price, rating, popularity
        hidden_dim: int = 256,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, item_emb_dim)
        self.category_embedding = nn.Embedding(num_categories, cat_emb_dim)

        # Project 384-d content embedding down to 64-d to reduce dominance
        self.content_projector = nn.Linear(content_emb_dim, 64)

        input_dim = item_emb_dim + cat_emb_dim + 64 + num_extra_features
        self.tower = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        item_ids: torch.Tensor,
        category_ids: torch.Tensor,
        content_embs: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            item_ids: (batch_size,) integer item indices
            category_ids: (batch_size,) integer category indices
            content_embs: (batch_size, 384) pre-computed content embeddings
            item_features: (batch_size, 3) float tensor [price, rating, popularity]

        Returns:
            (batch_size, 256) L2-normalized item embeddings
        """
        i_emb = self.item_embedding(item_ids)
        c_emb = self.category_embedding(category_ids)
        content_proj = self.content_projector(content_embs)
        x = torch.cat([i_emb, c_emb, content_proj, item_features], dim=-1)
        out = self.tower(x)
        return F.normalize(out, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """
    Complete Two-Tower model combining User and Item towers.

    Training: in-batch sampled softmax loss
    Inference: only run user tower, then ANN search
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_categories: int = 8,
        num_clusters: int = 20,
        output_dim: int = 256,
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        self.user_tower = UserTower(
            num_users=num_users,
            num_clusters=num_clusters,
            output_dim=output_dim,
        )
        self.item_tower = ItemTower(
            num_items=num_items,
            num_categories=num_categories,
            output_dim=output_dim,
        )
        # Temperature for softmax — lower = sharper, more discriminative
        self.temperature = temperature

    def forward(
        self,
        user_ids: torch.Tensor,
        cluster_ids: torch.Tensor,
        user_features: torch.Tensor,
        item_ids: torch.Tensor,
        category_ids: torch.Tensor,
        content_embs: torch.Tensor,
        item_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode users and items, return both embeddings.

        Returns: (user_embeddings, item_embeddings) each (batch_size, 256)
        """
        user_embs = self.user_tower(user_ids, cluster_ids, user_features)
        item_embs = self.item_tower(item_ids, category_ids, content_embs, item_features)
        return user_embs, item_embs

    def compute_loss(
        self,
        user_embs: torch.Tensor,
        item_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        In-batch sampled softmax loss.

        HOW IT WORKS:
          Given a batch of N (user, item) pairs:
          - Compute NxN similarity matrix: sim[i][j] = dot(user_i, item_j)
          - The diagonal (i==j) are positive pairs
          - Everything else is a negative pair
          - Apply cross-entropy: each row should have highest score on the diagonal

        WHY IT WORKS:
          This is extremely efficient — you get N positive and N*(N-1) negative
          pairs from a single batch of N. No explicit negative sampling needed.
          A batch of 256 gives you 256 positives and 65,280 negatives.

        TEMPERATURE:
          Dividing by temperature (0.05) makes the softmax sharper.
          Without it, the model might not learn to discriminate well enough.
        """
        # Similarity matrix: (batch, batch)
        logits = torch.matmul(user_embs, item_embs.T) / self.temperature

        # Labels: each user_i should match item_i (diagonal)
        labels = torch.arange(logits.shape[0], device=logits.device)

        # Cross-entropy loss (symmetric: user→item and item→user)
        loss_u2i = F.cross_entropy(logits, labels)
        loss_i2u = F.cross_entropy(logits.T, labels)

        return (loss_u2i + loss_i2u) / 2
