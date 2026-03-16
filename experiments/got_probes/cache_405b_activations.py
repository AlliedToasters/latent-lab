"""
Cache all 405B activations for GoT datasets via NDIF.

Extraction only — no probe fitting. Avoids OOM from assembling
the full feature matrix. Re-run until all prompts are cached.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from lmprobe import LinearProbe, enable_cache_logging, set_max_threads
from lmprobe.cache import set_cache_dtype, cache_info

set_max_threads(8)
set_cache_dtype("float16")
enable_cache_logging()

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_ID = "meta-llama/Llama-3.1-405B-Instruct"
RANDOM_STATE = 42
TEST_SIZE = 0.2

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


def main():
    import argparse
    import gc

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, choices=CURATED_DATASETS,
                        help="Single dataset (default: run all)")
    args = parser.parse_args()

    if not os.environ.get("NNSIGHT_API_KEY"):
        print("ERROR: NNSIGHT_API_KEY not set")
        return

    print(f"NNSIGHT_API_KEY: set")
    print(f"Disk status:")
    disk_status()
    print()

    datasets = [args.dataset] if args.dataset else CURATED_DATASETS

    for ds in datasets:
        prompts = get_all_prompts(ds)
        print(f"\n{'='*60}")
        print(f"Caching: {ds} ({len(prompts)} prompts)")
        print(f"{'='*60}")

        # Fresh probe per dataset to avoid memory accumulation
        probe = LinearProbe(
            model=MODEL_ID,
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

        del probe
        gc.collect()
        disk_status()

    print(f"\nFinal status:")
    disk_status()
    print(f"\n{cache_info()}")


if __name__ == "__main__":
    main()
