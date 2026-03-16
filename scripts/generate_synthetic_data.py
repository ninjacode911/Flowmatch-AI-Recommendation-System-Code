"""
Synthetic Data Generator for the FlowMatch Recommendation System.

Generates realistic fake data for 3 entities:
  1. Items    — products/content in the catalogue (50K)
  2. Users    — registered users with preferences (10K)
  3. Interactions — click/view/purchase events between users and items

WHY SYNTHETIC DATA?
  In a real company, you'd use production logs. For a portfolio project, we
  generate data that has the SAME statistical properties as real data:
    - Power-law item popularity (few items get most clicks — like real life)
    - User preference clusters (users in the same segment like similar categories)
    - Temporal patterns (recent items get more views)
    - Implicit feedback (clicks are common, purchases are rare)

  This lets us train models that would also work on real data.

Usage:
  python scripts/generate_synthetic_data.py
  python scripts/generate_synthetic_data.py --num-users 5000 --num-items 20000
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

# ─── Configuration ───────────────────────────────────────────────

# Item categories with subcategories (e-commerce + content hybrid)
CATEGORIES = {
    "electronics": ["smartphones", "laptops", "headphones", "tablets", "cameras", "smartwatches"],
    "clothing": ["t-shirts", "jeans", "jackets", "dresses", "shoes", "activewear"],
    "home": ["furniture", "kitchen", "decor", "lighting", "bedding", "storage"],
    "books": ["fiction", "non-fiction", "tech", "self-help", "biography", "science"],
    "sports": ["running", "cycling", "yoga", "weights", "swimming", "hiking"],
    "beauty": ["skincare", "makeup", "haircare", "fragrance", "tools", "wellness"],
    "food": ["snacks", "beverages", "organic", "supplements", "cooking", "international"],
    "toys": ["educational", "outdoor", "board-games", "building", "plush", "electronic"],
}

# Adjectives and nouns for generating realistic product titles
ADJECTIVES = [
    "Premium", "Classic", "Ultra", "Pro", "Essential", "Deluxe", "Compact",
    "Advanced", "Eco", "Smart", "Vintage", "Modern", "Lightweight", "Heavy-Duty",
    "Portable", "Wireless", "Organic", "Natural", "Elite", "Budget",
]

BRANDS = [
    "NovaTech", "PeakGear", "UrbanEdge", "ZenithCo", "PureForm", "SwiftLine",
    "EcoVibe", "LuxCraft", "CoreFit", "BrightPath", "ArcticWave", "SolarPeak",
    "VelvetStone", "IronForge", "CrystalClear", "ThunderBolt", "GreenLeaf", "OceanBreeze",
]

EVENT_TYPES = ["view", "click", "add_to_cart", "purchase"]
# Probabilities: most interactions are views, few are purchases (realistic funnel)
EVENT_WEIGHTS = [0.50, 0.30, 0.12, 0.08]

LOCATIONS = [
    "US-NY", "US-CA", "US-TX", "US-FL", "US-IL", "US-WA", "US-MA",
    "UK-LDN", "UK-MAN", "DE-BER", "DE-MUN", "FR-PAR", "IN-MUM",
    "IN-BLR", "IN-DEL", "JP-TKY", "AU-SYD", "CA-TOR", "BR-SAO",
]


def generate_items(num_items: int, rng: np.random.Generator) -> list[dict]:
    """
    Generate synthetic item catalogue.

    Each item has:
      - Unique ID, title, description, category, subcategory
      - Price (log-normal distribution — most items cheap, few expensive)
      - Rating (normal distribution centered at 3.8)
      - Popularity score (power-law — few items are very popular)
      - Creation timestamp (spread over last 365 days)
    """
    items = []
    now = time.time()
    one_year = 365 * 24 * 3600

    # Power-law popularity: rank-based with Zipf distribution
    # This means item #1 is ~2x more popular than #2, ~3x more than #3, etc.
    popularity_raw = 1.0 / np.arange(1, num_items + 1) ** 0.8
    popularity_scores = popularity_raw / popularity_raw.max()
    rng.shuffle(popularity_scores)

    categories = list(CATEGORIES.keys())

    for i in range(num_items):
        cat = categories[i % len(categories)]
        subcat = CATEGORIES[cat][i % len(CATEGORIES[cat])]
        adj = ADJECTIVES[rng.integers(len(ADJECTIVES))]
        brand = BRANDS[rng.integers(len(BRANDS))]

        # Log-normal price: most items $10-50, some up to $500+
        price = round(float(rng.lognormal(mean=3.0, sigma=0.8)), 2)
        price = min(price, 999.99)

        # Rating: normal around 3.8, clipped to [1, 5]
        rating = round(float(np.clip(rng.normal(3.8, 0.7), 1.0, 5.0)), 1)

        # Created within the last year
        created_at = now - rng.uniform(0, one_year)

        items.append({
            "item_id": f"item_{i:06d}",
            "title": f"{brand} {adj} {subcat.replace('-', ' ').title()}",
            "description": f"High-quality {subcat} from {brand}. Category: {cat}.",
            "category": cat,
            "subcategory": subcat,
            "tags": [cat, subcat, brand.lower(), adj.lower()],
            "price": price,
            "brand": brand,
            "rating": rating,
            "popularity_score": round(float(popularity_scores[i]), 4),
            "created_at": round(created_at, 2),
        })

    return items


def generate_users(num_users: int, rng: np.random.Generator) -> list[dict]:
    """
    Generate synthetic user profiles.

    Each user has:
      - Unique ID, age, gender, location
      - 2-4 preferred categories (this drives their interaction patterns)
      - A "taste cluster" ID (users in same cluster like similar things)

    WHY TASTE CLUSTERS?
      Real users form natural segments (e.g., "tech-savvy millennials",
      "budget-conscious parents"). Collaborative filtering learns these
      patterns. We simulate this by assigning users to clusters, where
      each cluster has preferred categories.
    """
    users = []
    categories = list(CATEGORIES.keys())
    num_clusters = 20  # 20 distinct user segments

    # Each cluster prefers 2-3 categories
    cluster_prefs = {}
    for c in range(num_clusters):
        n_cats = rng.integers(2, 4)
        cluster_prefs[c] = list(rng.choice(categories, size=n_cats, replace=False))

    for i in range(num_users):
        cluster_id = i % num_clusters
        age = int(np.clip(rng.normal(32, 10), 18, 70))
        gender = rng.choice(["M", "F", "NB"], p=[0.48, 0.48, 0.04])
        location = LOCATIONS[rng.integers(len(LOCATIONS))]

        users.append({
            "user_id": f"user_{i:06d}",
            "age": age,
            "gender": str(gender),
            "location": str(location),
            "preferred_categories": cluster_prefs[cluster_id],
            "cluster_id": cluster_id,
        })

    return users


def generate_interactions(
    users: list[dict],
    items: list[dict],
    num_interactions: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Generate synthetic user-item interactions (vectorized for speed).

    KEY DESIGN DECISIONS:
      1. Users interact more with items in their preferred categories (80% of the time)
         — this is what collaborative filtering will learn to predict.
      2. Popular items get more interactions (power-law) — mimics real platforms.
      3. Event types follow a funnel: view (50%) > click (30%) > add_to_cart (12%) > purchase (8%)
      4. Timestamps are spread over the last 90 days with a recency bias.

    The interaction data is what we'll use to train:
      - Two-Tower model (user clicked/purchased item = positive pair)
      - Collaborative filtering (user-item interaction matrix)
      - LTR model (features derived from these interactions)
    """
    now = time.time()
    ninety_days = 90 * 24 * 3600

    # Build category-to-items index as numpy arrays for fast sampling
    cat_to_items: dict[str, np.ndarray] = {}
    for idx, item in enumerate(items):
        cat = item["category"]
        if cat not in cat_to_items:
            cat_to_items[cat] = []
        cat_to_items[cat].append(idx)
    for cat in cat_to_items:
        cat_to_items[cat] = np.array(cat_to_items[cat])

    categories = list(cat_to_items.keys())

    # Item popularity weights for random selection
    pop_scores = np.array([item["popularity_score"] for item in items])
    pop_probs = pop_scores / pop_scores.sum()

    # Pre-extract user data into arrays for vectorized access
    user_ids_arr = [u["user_id"] for u in users]
    user_pref_cats = [u["preferred_categories"] for u in users]
    item_ids_arr = [it["item_id"] for it in items]

    print("  Generating interactions (vectorized)...")

    # Pre-generate all random numbers at once (MUCH faster than one-by-one)
    user_indices = rng.integers(0, len(users), size=num_interactions)
    use_preferred = rng.random(size=num_interactions) < 0.8
    event_indices = rng.choice(len(EVENT_TYPES), size=num_interactions, p=EVENT_WEIGHTS)
    recency_vals = np.minimum(rng.exponential(scale=0.3, size=num_interactions), 1.0)
    timestamps = np.round(now - recency_vals * ninety_days, 2)

    # Pre-generate popularity-weighted item choices for exploration interactions
    pop_item_indices = rng.choice(len(items), size=num_interactions, p=pop_probs)

    # For preferred-category interactions, pre-generate random indices
    # We'll pick a random preferred category per user, then a random item from that category
    pref_cat_random = rng.random(size=num_interactions)  # for choosing which preferred cat
    pref_item_random = rng.random(size=num_interactions)  # for choosing item within cat

    interactions = []
    for i in range(num_interactions):
        u_idx = user_indices[i]
        prefs = user_pref_cats[u_idx]

        if use_preferred[i] and prefs:
            cat_idx = int(pref_cat_random[i] * len(prefs)) % len(prefs)
            cat = prefs[cat_idx]
            cat_items = cat_to_items[cat]
            item_idx = cat_items[int(pref_item_random[i] * len(cat_items)) % len(cat_items)]
        else:
            item_idx = pop_item_indices[i]

        interactions.append({
            "user_id": user_ids_arr[u_idx],
            "item_id": item_ids_arr[item_idx],
            "event_type": EVENT_TYPES[event_indices[i]],
            "timestamp": float(timestamps[i]),
        })

        if (i + 1) % 500_000 == 0:
            print(f"    {i+1:,} / {num_interactions:,} interactions generated...")

    return interactions


def save_data(data: list[dict], filepath: Path) -> None:
    """Save data as JSON Lines format (one JSON object per line)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")
    print(f"  Saved {len(data):,} records to {filepath}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic data for RecSys")
    parser.add_argument("--num-users", type=int, default=50_000, help="Number of users")
    parser.add_argument("--num-items", type=int, default=50_000, help="Number of items")
    parser.add_argument("--num-interactions", type=int, default=2_000_000, help="Number of interactions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "synthetic"),
        help="Output directory",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)

    print(f"Generating synthetic data (seed={args.seed})...")
    print(f"  Users: {args.num_users:,}")
    print(f"  Items: {args.num_items:,}")
    print(f"  Interactions: {args.num_interactions:,}")
    print()

    # Generate
    items = generate_items(args.num_items, rng)
    users = generate_users(args.num_users, rng)
    interactions = generate_interactions(users, items, args.num_interactions, rng)

    # Save
    save_data(items, output_dir / "items.jsonl")
    save_data(users, output_dir / "users.jsonl")
    save_data(interactions, output_dir / "interactions.jsonl")

    # Print sample
    print("\n--- Sample Item ---")
    print(json.dumps(items[0], indent=2))
    print("\n--- Sample User ---")
    print(json.dumps(users[0], indent=2))
    print("\n--- Sample Interaction ---")
    print(json.dumps(interactions[0], indent=2))

    # Print stats
    from collections import Counter
    event_counts = Counter(ix["event_type"] for ix in interactions)
    print("\n--- Interaction Stats ---")
    for event, count in event_counts.most_common():
        print(f"  {event}: {count:,} ({count/len(interactions)*100:.1f}%)")

    cats_in_interactions = Counter(
        next(it["category"] for it in items if it["item_id"] == ix["item_id"])
        for ix in interactions[:1000]  # sample first 1K for speed
    )
    print("\n--- Category Distribution (sample of 1K interactions) ---")
    for cat, count in cats_in_interactions.most_common():
        print(f"  {cat}: {count}")

    print("\nDone!")


if __name__ == "__main__":
    main()
