"""
NCF (Neural Collaborative Filtering) Training Script.

=== KEY DIFFERENCE FROM TWO-TOWER ===

Two-Tower uses in-batch negatives (efficient, no explicit negative sampling).
NCF uses EXPLICIT negative sampling: for each positive (user, item) pair,
we generate N random items the user hasn't interacted with as negatives.

WHY EXPLICIT NEGATIVES?
  NCF produces a scalar score (0-1), not an embedding for ANN search.
  It needs to see both positive AND negative examples to learn the
  decision boundary. The model must learn:
    - "This user-item pair has score 0.95" (positive)
    - "This user-item pair has score 0.05" (negative)

NEGATIVE RATIO:
  We use 4 negatives per positive. More negatives = better calibration
  but slower training. 4:1 is a standard ratio from the NCF paper.

Usage:
  python -m services.training_pipeline.app.train_ncf
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from services.training_pipeline.app.models.ncf import NeuMF

# Auto-detect GPU (RTX 5070 works in WSL2 with PyTorch 2.7+)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Hyperparameters ──────────────────────────────────────────────────────────

BATCH_SIZE = 2048         # Larger batch for GPU throughput
EPOCHS = 100             # Train up to 100 epochs — early stopping will halt when converged
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3      # Stronger regularization to fight overfitting (was 1e-5)
NEG_RATIO = 6            # 6 negatives per positive (stronger discrimination signal)
VAL_SPLIT = 0.1
GMF_EMB_DIM = 64
MLP_EMB_DIM = 64
MLP_HIDDEN = [256, 128, 64]  # Deeper MLP for more data (was [128, 64, 32])
EARLY_STOP_PATIENCE = 10 # Stop if no val_loss improvement for 10 epochs
USE_AMP = DEVICE.type == "cuda"

DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"


class NCFDataset(Dataset):
    """
    Dataset for NCF training with explicit negative sampling.

    Each epoch re-samples negatives so the model sees different negatives
    each time -- this improves generalization.
    """

    def __init__(
        self,
        interactions_path: str,
        users_path: str,
        num_items: int,
        event_types: list[str] | None = None,
        neg_ratio: int = 4,
    ) -> None:
        if event_types is None:
            event_types = ["click", "add_to_cart", "purchase"]

        # Load user ID mapping
        self.user_id_to_idx: dict[str, int] = {}
        with open(users_path) as f:
            for line in f:
                user = json.loads(line)
                self.user_id_to_idx[user["user_id"]] = len(self.user_id_to_idx)

        self.num_users = len(self.user_id_to_idx)
        self.num_items = num_items
        self.neg_ratio = neg_ratio

        # Build positive interactions set
        self.positives: list[tuple[int, int]] = []
        self.user_items: dict[int, set[int]] = {}  # For negative sampling

        with open(interactions_path) as f:
            for line in f:
                ix = json.loads(line)
                if ix["event_type"] in event_types:
                    uid = ix["user_id"]
                    iid = ix["item_id"]
                    if uid in self.user_id_to_idx:
                        u_idx = self.user_id_to_idx[uid]
                        # item_id format is "item_XXXXXX", extract index
                        i_idx = int(iid.split("_")[1])
                        if i_idx < num_items:
                            self.positives.append((u_idx, i_idx))
                            if u_idx not in self.user_items:
                                self.user_items[u_idx] = set()
                            self.user_items[u_idx].add(i_idx)

        # Deduplicate positives
        self.positives = list(set(self.positives))

        # Generate initial negatives
        self.samples: list[tuple[int, int, float]] = []
        self._sample_negatives()

        print(f"NCF Dataset: {len(self.positives):,} positive pairs")
        print(f"  With {neg_ratio}x negatives: {len(self.samples):,} total samples per epoch")

    def _sample_negatives(self) -> None:
        """Re-sample negatives for a new epoch."""
        self.samples = []
        rng = np.random.default_rng()

        for u_idx, i_idx in self.positives:
            # Add positive
            self.samples.append((u_idx, i_idx, 1.0))

            # Sample negatives
            user_pos = self.user_items.get(u_idx, set())
            neg_count = 0
            while neg_count < self.neg_ratio:
                neg_item = rng.integers(0, self.num_items)
                if neg_item not in user_pos:
                    self.samples.append((u_idx, int(neg_item), 0.0))
                    neg_count += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        u_idx, i_idx, label = self.samples[idx]
        return {
            "user_id": torch.tensor(u_idx, dtype=torch.long),
            "item_id": torch.tensor(i_idx, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
        }


def train_one_epoch(
    model: NeuMF,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        user_ids = batch["user_id"].to(DEVICE, non_blocking=True)
        item_ids = batch["item_id"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            predictions = model(user_ids, item_ids)
            loss = model.compute_loss(predictions, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 100 == 0:
            avg = total_loss / num_batches
            print(f"  Epoch {epoch+1} [{batch_idx+1}/{len(loader)}] loss={avg:.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model: NeuMF, loader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        user_ids = batch["user_id"].to(DEVICE)
        item_ids = batch["item_id"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        predictions = model(user_ids, item_ids)
        loss = model.compute_loss(predictions, labels)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main() -> None:
    print("=" * 60)
    print("NCF (NEURAL COLLABORATIVE FILTERING) TRAINING")
    print("=" * 60)

    # Count items
    num_items = 0
    with open(DATA_DIR / "items.jsonl") as f:
        for _ in f:
            num_items += 1
    print(f"\nTotal items: {num_items:,}")

    # Load dataset
    print("\n[1/4] Loading dataset...")
    dataset = NCFDataset(
        interactions_path=str(DATA_DIR / "interactions.jsonl"),
        users_path=str(DATA_DIR / "users.jsonl"),
        num_items=num_items,
        neg_ratio=NEG_RATIO,
    )

    # Split
    print("\n[2/4] Creating data loaders...")
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    is_gpu = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4 if is_gpu else 0, drop_last=True,
        pin_memory=is_gpu, persistent_workers=True if is_gpu else False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4 if is_gpu else 0,
        pin_memory=is_gpu, persistent_workers=True if is_gpu else False,
    )

    print(f"Train: {train_size:,} samples -> {len(train_loader):,} batches")
    print(f"Val:   {val_size:,} samples -> {len(val_loader):,} batches")

    # Model
    print("\n[3/4] Initializing model...")
    model = NeuMF(
        num_users=dataset.num_users,
        num_items=num_items,
        gmf_emb_dim=GMF_EMB_DIM,
        mlp_emb_dim=MLP_EMB_DIM,
        mlp_hidden_layers=MLP_HIDDEN,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    # Train
    print(f"\n[4/4] Training for up to {EPOCHS} epochs (early stopping patience={EARLY_STOP_PATIENCE})...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Neg ratio: {NEG_RATIO}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Device: {DEVICE}")
    print(f"  Mixed Precision (AMP): {USE_AMP}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        # Re-sample negatives each epoch for diversity
        if epoch > 0:
            dataset._sample_negatives()

        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch, scaler)
        val_loss = validate(model, val_loader)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "num_users": dataset.num_users,
                    "num_items": num_items,
                },
                MODEL_DIR / "ncf_best.pt",
            )
            improved = " * (saved)"
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch+1:>2}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"lr={lr:.6f} | "
            f"{epoch_time:.1f}s{improved}"
        )

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping: no improvement for {EARLY_STOP_PATIENCE} epochs")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s ({epoch+1} epochs)")
    print(f"Best validation loss: {best_val_loss:.4f}")

    print("\n" + "=" * 60)
    print("NCF TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model checkpoint: {MODEL_DIR / 'ncf_best.pt'}")
    print(f"\nNCF will be used as a scoring model in the ranking stage.")
    print("Next: LightGBM LTR ranker with engineered features.")


if __name__ == "__main__":
    main()
