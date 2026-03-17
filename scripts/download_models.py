"""
Download model artifacts and synthetic data from GitHub Releases.

This script is used by Render (or any cloud host) during the build step
to fetch the large files that are git-ignored.
"""

import os
import urllib.request
import sys
from pathlib import Path

REPO = "ninjacode911/Navnit-s-Flowmatch-AI-Recommendation-System"
TAG = "v1.0-models"
BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}"

FILES = [
    # (remote_filename, local_path)
    ("two_tower_best.pt", "models/artifacts/two_tower_best.pt"),
    ("two_tower_item_embeddings.npy", "models/artifacts/two_tower_item_embeddings.npy"),
    ("two_tower_id_to_idx.json", "models/artifacts/two_tower_id_to_idx.json"),
    ("ltr_lightgbm.txt", "models/artifacts/ltr_lightgbm.txt"),
    ("users.jsonl", "data/synthetic/users.jsonl"),
    ("items.jsonl", "data/synthetic/items.jsonl"),
    ("interactions.jsonl", "data/synthetic/interactions.jsonl"),
]


def download_file(url: str, dest: str) -> None:
    dest_path = Path(dest)
    if dest_path.exists():
        print(f"  SKIP (exists): {dest}")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {url}")
    print(f"  -> {dest}")

    urllib.request.urlretrieve(url, str(dest_path))

    size_mb = dest_path.stat().st_size / (1024 * 1024)
    print(f"  Done: {size_mb:.1f} MB")


def main() -> None:
    print("=" * 60)
    print("Downloading model artifacts and data...")
    print("=" * 60)

    for remote_name, local_path in FILES:
        url = f"{BASE_URL}/{remote_name}"
        try:
            download_file(url, local_path)
        except Exception as e:
            print(f"  ERROR downloading {remote_name}: {e}", file=sys.stderr)
            sys.exit(1)

    print("\nAll files downloaded successfully!")


if __name__ == "__main__":
    main()
