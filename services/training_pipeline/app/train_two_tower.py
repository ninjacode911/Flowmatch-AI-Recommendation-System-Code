"""
Two-Tower Model Training Script

=== TRAINING OVERVIEW ===

This script orchestrates the full training loop:

1. Load Data: InteractionDataset provides (user, item) positive pairs
2. DataLoader: Shuffled batches of 512 pairs
3. Forward Pass: Both towers encode their inputs → 256-d embeddings
4. Loss: In-batch sampled softmax (each batch gives 512 positives + ~262K negatives)
5. Backward Pass: AdamW optimizer with cosine LR schedule
6. Export: Save trained item embeddings for Qdrant indexing

=== KEY TRAINING DECISIONS ===

- AdamW over Adam: decouples weight decay from the gradient update,
  leading to better generalization. Standard for modern deep learning.

- Cosine LR Schedule: starts at peak LR, smoothly decays to near-zero.
  Avoids the need to manually pick step milestones. Works well for
  fixed-epoch training.

- Mixed Precision (AMP): uses float16 for forward/backward passes,
  float32 for weight updates. ~2x faster on GPU, same accuracy.
  On CPU, we skip AMP since there's no speedup.

- Gradient Clipping (max_norm=1.0): prevents exploding gradients,
  especially important with the low temperature (0.05) in the loss.

Usage:
  python -m services.training_pipeline.app.train_two_tower
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from services.training_pipeline.app.dataset import InteractionDataset
from services.training_pipeline.app.models.two_tower import TwoTowerModel

# Auto-detect GPU (RTX 5070 works in WSL2 with PyTorch 2.7+)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Hyperparameters ──────────────────────────────────────────────────────────

BATCH_SIZE = 1024         # Larger batch = more in-batch negatives (1024x1023 = ~1M negatives/batch)
EPOCHS = 100             # Train up to 100 epochs — early stopping will halt when converged
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
VAL_SPLIT = 0.1          # 10% of data for validation
TEMPERATURE = 0.05
OUTPUT_DIM = 256
GRAD_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 12 # Stop if no val_loss improvement for 12 epochs
USE_AMP = DEVICE.type == "cuda"  # Mixed precision on GPU for ~2x speedup

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"


def create_dataloaders(
    dataset: InteractionDataset,
    batch_size: int,
    val_split: float,
) -> tuple[DataLoader, DataLoader]:
    """
    Split dataset into train/val and create DataLoaders.

    WHY VALIDATE?
      We need to check that the model generalizes to unseen (user, item) pairs.
      If train loss drops but val loss doesn't, we're overfitting — the model
      is memorizing specific pairs instead of learning general preferences.
    """
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    is_gpu = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,       # CRITICAL: shuffle so each batch has diverse negatives
        num_workers=4 if is_gpu else 0,  # Multi-process loading on Linux/WSL
        drop_last=True,      # Drop incomplete final batch (in-batch negatives need full batches)
        pin_memory=is_gpu,   # Pin memory for faster CPU→GPU transfer
        persistent_workers=True if is_gpu else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if is_gpu else 0,
        drop_last=True,
        pin_memory=is_gpu,
        persistent_workers=True if is_gpu else False,
    )

    print(f"Train: {train_size:,} pairs -> {len(train_loader):,} batches")
    print(f"Val:   {val_size:,} pairs -> {len(val_loader):,} batches")

    return train_loader, val_loader


def train_one_epoch(
    model: TwoTowerModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Run one training epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        # Move tensors to device (non_blocking for pinned memory)
        user_ids = batch["user_id"].to(DEVICE, non_blocking=True)
        cluster_ids = batch["cluster_id"].to(DEVICE, non_blocking=True)
        user_features = batch["user_features"].to(DEVICE, non_blocking=True)
        item_ids = batch["item_id"].to(DEVICE, non_blocking=True)
        category_ids = batch["category_id"].to(DEVICE, non_blocking=True)
        content_embs = batch["content_emb"].to(DEVICE, non_blocking=True)
        item_features = batch["item_features"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass with optional AMP (mixed precision)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            user_embs, item_embs = model(
                user_ids, cluster_ids, user_features,
                item_ids, category_ids, content_embs, item_features,
            )
            loss = model.compute_loss(user_embs, item_embs)

        # Backward pass with gradient scaling for AMP
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Log progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            avg_so_far = total_loss / num_batches
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1} [{batch_idx+1}/{len(loader)}] loss={avg_so_far:.4f} lr={lr:.6f}")

    # Step the LR scheduler once per epoch
    scheduler.step()

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model: TwoTowerModel, loader: DataLoader) -> float:
    """Run validation, return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        user_ids = batch["user_id"].to(DEVICE)
        cluster_ids = batch["cluster_id"].to(DEVICE)
        user_features = batch["user_features"].to(DEVICE)
        item_ids = batch["item_id"].to(DEVICE)
        category_ids = batch["category_id"].to(DEVICE)
        content_embs = batch["content_emb"].to(DEVICE)
        item_features = batch["item_features"].to(DEVICE)

        user_embs, item_embs = model(
            user_ids, cluster_ids, user_features,
            item_ids, category_ids, content_embs, item_features,
        )
        loss = model.compute_loss(user_embs, item_embs)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def export_item_embeddings(
    model: TwoTowerModel,
    dataset: InteractionDataset,
    output_dir: Path,
) -> None:
    """
    Pre-compute item embeddings from the trained Item Tower.

    WHY PRE-COMPUTE?
      At inference time, we only run the User Tower (fast, ~5ms).
      The item embeddings are stored in Qdrant for ANN search.
      This is the key insight that makes Two-Tower scalable:
        - 50K items × 256-d = ~50MB in Qdrant (tiny!)
        - One user embedding + ANN search = ~10ms total
    """
    model.eval()
    print("\nExporting item embeddings from trained Item Tower...")

    num_items = len(dataset.item_id_to_idx)
    embeddings = np.zeros((num_items, OUTPUT_DIM), dtype=np.float32)

    # Process all items in batches
    batch_size = 1024
    item_ids_sorted = sorted(dataset.item_id_to_idx.keys(), key=lambda x: dataset.item_id_to_idx[x])

    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        batch_item_ids = item_ids_sorted[start:end]

        # Build tensors for this batch of items
        idx_list = []
        cat_list = []
        content_list = []
        feat_list = []

        for item_id in batch_item_ids:
            item = dataset.items[item_id]
            idx = dataset.item_id_to_idx[item_id]
            idx_list.append(idx)
            cat_list.append(dataset.category_to_idx[item["category"]])
            content_list.append(dataset.content_embeddings[idx])

            price_norm = (item.get("price", 0) - dataset.price_mean) / dataset.price_std
            rating_norm = (item.get("rating", 3.0) - 3.0) / 1.0
            pop_score = item.get("popularity_score", 0.0)
            feat_list.append([price_norm, rating_norm, pop_score])

        item_ids_t = torch.tensor(idx_list, dtype=torch.long, device=DEVICE)
        cat_ids_t = torch.tensor(cat_list, dtype=torch.long, device=DEVICE)
        content_t = torch.tensor(np.array(content_list), dtype=torch.float32, device=DEVICE)
        feats_t = torch.tensor(feat_list, dtype=torch.float32, device=DEVICE)

        item_embs = model.item_tower(item_ids_t, cat_ids_t, content_t, feats_t)
        embeddings[start:end] = item_embs.cpu().numpy()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = output_dir / "two_tower_item_embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Saved item embeddings: {emb_path} shape={embeddings.shape}")

    # Also save the id-to-index mapping
    import json
    map_path = output_dir / "two_tower_id_to_idx.json"
    with open(map_path, "w") as f:
        json.dump(dataset.item_id_to_idx, f)
    print(f"Saved id mapping: {map_path}")


def main() -> None:
    print("=" * 60)
    print("TWO-TOWER MODEL TRAINING")
    print("=" * 60)

    # ── 1. Load dataset ──────────────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    dataset = InteractionDataset(
        interactions_path=str(DATA_DIR / "interactions.jsonl"),
        users_path=str(DATA_DIR / "users.jsonl"),
        items_path=str(DATA_DIR / "items.jsonl"),
        content_embeddings_path=str(DATA_DIR / "item_embeddings.npy"),
    )

    # ── 2. Create DataLoaders ────────────────────────────────────────────
    print("\n[2/5] Creating data loaders...")
    train_loader, val_loader = create_dataloaders(dataset, BATCH_SIZE, VAL_SPLIT)

    # ── 3. Initialize model ──────────────────────────────────────────────
    print("\n[3/5] Initializing model...")
    model = TwoTowerModel(
        num_users=len(dataset.user_id_to_idx),
        num_items=len(dataset.item_id_to_idx),
        num_categories=len(dataset.category_to_idx),
        num_clusters=20,
        output_dim=OUTPUT_DIM,
        temperature=TEMPERATURE,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"User Tower: {sum(p.numel() for p in model.user_tower.parameters()):,}")
    print(f"Item Tower: {sum(p.numel() for p in model.item_tower.parameters()):,}")

    # ── 4. Set up optimizer and scheduler ────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Cosine annealing: LR decays smoothly from LEARNING_RATE to ~0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS - WARMUP_EPOCHS,
        eta_min=1e-6,
    )

    # ── 5. Training loop ─────────────────────────────────────────────────
    print(f"\n[4/5] Training for up to {EPOCHS} epochs (early stopping patience={EARLY_STOP_PATIENCE})...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Device: {DEVICE}")
    print(f"  Mixed Precision (AMP): {USE_AMP}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # AMP gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, scaler)
        val_loss = validate(model, val_loader)

        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model checkpoint
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                MODEL_DIR / "two_tower_best.pt",
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

    # ── 6. Export item embeddings ────────────────────────────────────────
    print("\n[5/5] Exporting item embeddings...")

    # Load best checkpoint before export
    checkpoint = torch.load(MODEL_DIR / "two_tower_best.pt", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

    export_item_embeddings(model, dataset, MODEL_DIR)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model checkpoint: {MODEL_DIR / 'two_tower_best.pt'}")
    print(f"  Item embeddings:  {MODEL_DIR / 'two_tower_item_embeddings.npy'}")
    print(f"  ID mapping:       {MODEL_DIR / 'two_tower_id_to_idx.json'}")
    print(f"\nNext step: Run evaluation with Two-Tower embeddings")


if __name__ == "__main__":
    main()
