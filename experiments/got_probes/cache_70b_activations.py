"""
Cache Llama 3.1 70B activations for GoT datasets via NDIF.

Caches BOTH base and instruct versions for direct comparison of
truth representations before/after instruct fine-tuning.

70B has 80 layers, 8192 hidden dim — much smaller than 405B.
Two models × 80 layers × 7660 prompts ≈ ~90GB each (estimate).

Usage:
    # Cache both models (default)
    python scripts/cache_70b_activations.py

    # Cache one model at a time
    python scripts/cache_70b_activations.py --model base
    python scripts/cache_70b_activations.py --model instruct

    # Single dataset
    python scripts/cache_70b_activations.py --model base --dataset cities
"""

import gc
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from lmprobe import LinearProbe, enable_cache_logging, set_max_threads
from lmprobe.cache import set_cache_dtype, set_cache_limit, cache_info

set_max_threads(8)
set_cache_dtype("float16")
set_cache_limit(None)  # Disable LRU eviction — we have 2TB+ free
enable_cache_logging()

DATA_DIR = Path(__file__).parent.parent / "data"
RANDOM_STATE = 42
TEST_SIZE = 0.2

MODELS = {
    "base": "meta-llama/Llama-3.1-70B",
    "instruct": "meta-llama/Llama-3.1-70B-Instruct",
}

CURATED_DATASETS = [
    "cities",
    "neg_cities",
    "sp_en_trans",
    "neg_sp_en_trans",
    "larger_than",
    "smaller_than",
]


def disk_status():
    cache_size = subprocess.run(
        ["du", "-sh", os.path.expanduser("~/.cache/lmprobe/")],
        capture_output=True, text=True
    ).stdout.strip()
    disk_free = subprocess.run(
        ["df", "-h", "/"], capture_output=True, text=True
    ).stdout.strip().split("\n")[-1]
    print(f"  Cache: {cache_size}")
    print(f"  Disk:  {disk_free}")


def get_all_prompts(dataset_name: str) -> list[str]:
    """Get all unique prompts (train + test) for a dataset."""
    df = pd.read_csv(DATA_DIR / f"{dataset_name}.csv")
    true_stmts = df[df["label"] == 1]["statement"].tolist()
    false_stmts = df[df["label"] == 0]["statement"].tolist()

    true_train, true_test = train_test_split(
        true_stmts, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    false_train, false_test = train_test_split(
        false_stmts, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return true_train + false_train + true_test + false_test


def cache_model(model_key: str, datasets: list[str]):
    """Cache all activations for one model across specified datasets."""
    model_id = MODELS[model_key]
    print(f"\n{'#'*60}")
    print(f"# Caching: {model_id}")
    print(f"# ({model_key} model, {len(datasets)} datasets)")
    print(f"{'#'*60}")

    for ds in datasets:
        prompts = get_all_prompts(ds)
        print(f"\n{'='*60}")
        print(f"  {model_key} / {ds} ({len(prompts)} prompts)")
        print(f"{'='*60}")

        probe = LinearProbe(
            model=model_id,
            layers="all",
            classifier="mass_mean",
            remote=True,
            backend="nnsight",
            random_state=RANDOM_STATE,
        )

        try:
            probe.warmup(prompts, remote=True)
            print(f"  -> {ds}: ALL CACHED")
        except Exception as e:
            print(f"  -> {ds}: FAILED ({e})")
            import traceback
            traceback.print_exc()

        del probe
        gc.collect()
        disk_status()

    print(f"\n  {model_key} caching complete.")
    disk_status()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cache 70B base/instruct activations for GoT datasets"
    )
    parser.add_argument(
        "--model", choices=["base", "instruct", "both"], default="both",
        help="Which model to cache (default: both)"
    )
    parser.add_argument(
        "--dataset", default=None, choices=CURATED_DATASETS,
        help="Single dataset (default: all)"
    )
    args = parser.parse_args()

    if not os.environ.get("NNSIGHT_API_KEY"):
        print("ERROR: NNSIGHT_API_KEY not set")
        print("  export NNSIGHT_API_KEY=<your key>")
        sys.exit(1)

    print(f"NNSIGHT_API_KEY: set")
    print(f"Initial disk status:")
    disk_status()

    datasets = [args.dataset] if args.dataset else CURATED_DATASETS

    if args.model in ("base", "both"):
        cache_model("base", datasets)

    if args.model in ("instruct", "both"):
        cache_model("instruct", datasets)

    print(f"\n{'='*60}")
    print(f"ALL DONE")
    print(f"{'='*60}")
    disk_status()
    print(f"\n{cache_info()}")


if __name__ == "__main__":
    main()
