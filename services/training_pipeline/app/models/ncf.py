"""
Neural Collaborative Filtering (NCF) -- learns user-item interaction patterns.

=== WHAT IS NCF? ===

NCF combines two complementary approaches to model user-item interactions:

1. GMF (Generalized Matrix Factorization):
   - Takes user embedding and item embedding
   - Computes element-wise product: user_emb * item_emb
   - This is like classic matrix factorization (SVD) but with learned embeddings
   - Captures LINEAR interaction patterns

2. MLP (Multi-Layer Perceptron):
   - Concatenates user and item embeddings
   - Passes through deep layers with ReLU
   - Captures NON-LINEAR interaction patterns
   - Can learn complex relationships that GMF misses

The NeuMF (Neural Matrix Factorization) model fuses both:
   output = sigmoid(W * [GMF_output | MLP_output])

=== WHY NCF ALONGSIDE TWO-TOWER? ===

Two-Tower is optimized for RETRIEVAL (fast ANN search from 50K items).
NCF is optimized for SCORING (accurate relevance prediction for ~500 candidates).

In the pipeline:
  1. Two-Tower retrieves top-500 candidates (fast, approximate)
  2. NCF scores those 500 candidates (slower, more accurate)
  3. LightGBM re-ranks with additional features (price, freshness, etc.)

=== ARCHITECTURE ===

User: user_id -> Embedding(64) -+-> element-wise product -> Linear -> |
                                |                                      | -> concat -> Linear -> sigmoid -> score
Item: item_id -> Embedding(64) -+-> concat -> MLP(128->64->32) -----> |

Reference: He et al., "Neural Collaborative Filtering" (WWW 2017)
"""

import torch
import torch.nn as nn


class GMF(nn.Module):
    """
    Generalized Matrix Factorization component.

    Like classic matrix factorization but with neural network embeddings.
    The element-wise product captures how each latent factor of the user
    aligns with the corresponding factor of the item.
    """

    def __init__(self, num_users: int, num_items: int, emb_dim: int = 64) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)

        # Initialize with small values (Xavier uniform)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_ids: (batch_size,) user indices
            item_ids: (batch_size,) item indices

        Returns:
            (batch_size, emb_dim) element-wise product
        """
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        return u_emb * i_emb  # Element-wise product


class MLP(nn.Module):
    """
    Multi-Layer Perceptron component.

    Concatenates user and item embeddings and passes through deep layers.
    This can capture non-linear interaction patterns that GMF misses.

    For example: "users who buy expensive electronics also buy premium cables"
    -- this cross-price-category pattern is non-linear and MLP can learn it.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        emb_dim: int = 64,
        hidden_layers: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Build MLP layers
        layers: list[nn.Module] = []
        input_dim = emb_dim * 2  # concat of user + item
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_layers[-1]

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_ids: (batch_size,) user indices
            item_ids: (batch_size,) item indices

        Returns:
            (batch_size, output_dim) MLP output
        """
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        x = torch.cat([u_emb, i_emb], dim=-1)
        return self.mlp(x)


class NeuMF(nn.Module):
    """
    Neural Matrix Factorization -- fuses GMF + MLP for final prediction.

    This is the full NCF model. It uses SEPARATE embeddings for GMF and MLP
    (so they can learn different representations), then concatenates their
    outputs and passes through a final prediction layer.

    Training uses binary cross-entropy:
      - Positive pairs (user interacted with item): label = 1
      - Negative pairs (random item user didn't interact with): label = 0
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        gmf_emb_dim: int = 64,
        mlp_emb_dim: int = 64,
        mlp_hidden_layers: list[int] | None = None,
    ) -> None:
        super().__init__()
        if mlp_hidden_layers is None:
            mlp_hidden_layers = [128, 64, 32]

        self.gmf = GMF(num_users, num_items, gmf_emb_dim)
        self.mlp = MLP(num_users, num_items, mlp_emb_dim, mlp_hidden_layers)

        # Final prediction layer: fuses GMF + MLP outputs (outputs raw logits)
        self.predict = nn.Linear(gmf_emb_dim + self.mlp.output_dim, 1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict interaction logits for (user, item) pairs.

        Args:
            user_ids: (batch_size,) user indices
            item_ids: (batch_size,) item indices

        Returns:
            (batch_size, 1) raw logits (apply sigmoid for probabilities)
        """
        gmf_out = self.gmf(user_ids, item_ids)
        mlp_out = self.mlp(user_ids, item_ids)
        x = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.predict(x)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Binary cross-entropy with logits loss (AMP-safe).

        Unlike Two-Tower (which uses in-batch negatives), NCF needs explicit
        negative sampling: for each positive (user, item) pair, we sample
        random items the user hasn't interacted with as negatives.
        """
        return nn.functional.binary_cross_entropy_with_logits(predictions.squeeze(), labels.float())
