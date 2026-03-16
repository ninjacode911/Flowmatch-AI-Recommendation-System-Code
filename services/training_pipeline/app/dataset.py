"""
Training Dataset for the Two-Tower model.

Loads interaction data and prepares (user, item) pairs for training.
Each pair represents "this user interacted with this item" (positive signal).

The in-batch negative sampling in the loss function handles negatives
automatically — we only need to provide positive pairs here.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


GENDER_MAP = {"M": 0, "F": 1, "NB": 2}


class InteractionDataset(Dataset):
    """
    PyTorch Dataset of (user_features, item_features) positive pairs.

    Each __getitem__ returns all the tensors needed by TwoTowerModel.forward().
    """

    def __init__(
        self,
        interactions_path: str,
        users_path: str,
        items_path: str,
        content_embeddings_path: str,
        event_types: list[str] | None = None,
    ) -> None:
        """
        Args:
            interactions_path: path to interactions.jsonl
            users_path: path to users.jsonl
            items_path: path to items.jsonl
            content_embeddings_path: path to item_embeddings.npy
            event_types: which events count as positive (default: click + purchase)
        """
        if event_types is None:
            event_types = ["click", "add_to_cart", "purchase"]

        # Load users
        self.users: dict[str, dict] = {}
        self.user_id_to_idx: dict[str, int] = {}
        with open(users_path) as f:
            for line in f:
                user = json.loads(line)
                idx = len(self.user_id_to_idx)
                self.user_id_to_idx[user["user_id"]] = idx
                self.users[user["user_id"]] = user

        # Load items
        self.items: dict[str, dict] = {}
        self.item_id_to_idx: dict[str, int] = {}
        self.categories: list[str] = []
        self.category_to_idx: dict[str, int] = {}
        with open(items_path) as f:
            for line in f:
                item = json.loads(line)
                idx = len(self.item_id_to_idx)
                self.item_id_to_idx[item["item_id"]] = idx
                self.items[item["item_id"]] = item
                cat = item["category"]
                if cat not in self.category_to_idx:
                    self.category_to_idx[cat] = len(self.category_to_idx)
                    self.categories.append(cat)

        # Load content embeddings
        self.content_embeddings = np.load(content_embeddings_path).astype(np.float32)

        # Compute normalization stats for numerical features
        prices = [item["price"] for item in self.items.values()]
        self.price_mean = np.mean(prices)
        self.price_std = np.std(prices) + 1e-8

        # Load interactions (positive pairs only)
        self.pairs: list[tuple[str, str]] = []
        with open(interactions_path) as f:
            for line in f:
                ix = json.loads(line)
                if ix["event_type"] in event_types:
                    uid, iid = ix["user_id"], ix["item_id"]
                    if uid in self.user_id_to_idx and iid in self.item_id_to_idx:
                        self.pairs.append((uid, iid))

        # Deduplicate: same (user, item) pair only counted once
        self.pairs = list(set(self.pairs))

        print(f"Dataset: {len(self.pairs):,} unique positive pairs")
        print(f"  Users: {len(self.user_id_to_idx):,}")
        print(f"  Items: {len(self.item_id_to_idx):,}")
        print(f"  Categories: {len(self.category_to_idx)}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        user_id, item_id = self.pairs[idx]

        user = self.users[user_id]
        item = self.items[item_id]

        # User features
        user_idx = self.user_id_to_idx[user_id]
        cluster_id = user.get("cluster_id", 0)
        age_norm = (user.get("age", 30) - 30) / 15.0  # Normalize around mean=30, std=15
        gender = user.get("gender", "M")
        gender_onehot = [
            1.0 if gender == "M" else 0.0,
            1.0 if gender == "F" else 0.0,
            1.0 if gender == "NB" else 0.0,
        ]

        # Item features
        item_idx = self.item_id_to_idx[item_id]
        cat_idx = self.category_to_idx[item["category"]]
        content_emb = self.content_embeddings[item_idx]
        price_norm = (item.get("price", 0) - self.price_mean) / self.price_std
        rating_norm = (item.get("rating", 3.0) - 3.0) / 1.0  # Center at 3.0
        pop_score = item.get("popularity_score", 0.0)

        return {
            "user_id": torch.tensor(user_idx, dtype=torch.long),
            "cluster_id": torch.tensor(cluster_id, dtype=torch.long),
            "user_features": torch.tensor([age_norm] + gender_onehot, dtype=torch.float32),
            "item_id": torch.tensor(item_idx, dtype=torch.long),
            "category_id": torch.tensor(cat_idx, dtype=torch.long),
            "content_emb": torch.tensor(content_emb, dtype=torch.float32),
            "item_features": torch.tensor([price_norm, rating_norm, pop_score], dtype=torch.float32),
        }
