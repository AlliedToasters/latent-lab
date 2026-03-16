"""
Train truth/falsehood probes on Geometry of Truth datasets using Llama 3.1 405B.

Remote execution via NDIF/nnsight — no local GPU needed.
Caches pooled (last_token) activations only to save disk.
"""

import json
import os
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from lmprobe import LinearProbe, enable_cache_logging, set_max_threads
from lmprobe.cache import set_cache_dtype, cache_info

# Use available CPU threads since no local GPU work
set_max_threads(8)

# Cache: float16 storage, limit controlled by LMPROBE_CACHE_MAX_GB env var
set_cache_dtype("float16")
enable_cache_logging()

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "metrics"

MODEL_ID = "meta-llama/Llama-3.1-405B-Instruct"
MODEL_SHORT = "llama3.1-405b"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# All curated GoT datasets
CURATED_DATASETS = [
    "cities",
    "neg_cities",
    "sp_en_trans",
    "neg_sp_en_trans",
    "larger_than",
    "smaller_than",
]

CLASSIFIERS = ["logistic_regression", "mass_mean"]


def disk_status():
    """Print current cache and disk usage."""
    cache_size = subprocess.run(
        ["du", "-sh", os.path.expanduser("~/.cache/lmprobe/")],
        capture_output=True, text=True
    ).stdout.strip()
    disk_free = subprocess.run(
        ["df", "-h", "/"], capture_output=True, text=True
    ).stdout.strip().split("\n")[-1]
    print(f"  Cache: {cache_size}")
    print(f"  Disk:  {disk_free}")


def load_dataset(name: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    true_stmts = df[df["label"] == 1]["statement"].tolist()
    false_stmts = df[df["label"] == 0]["statement"].tolist()
    return true_stmts, false_stmts


def train_and_evaluate(
    dataset_name: str,
    classifier: str,
    layers: str = "all",
) -> dict:
    true_stmts, false_stmts = load_dataset(dataset_name)

    true_train, true_test = train_test_split(
        true_stmts, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    false_train, false_test = train_test_split(
        false_stmts, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"\n{'='*60}")
    print(f"Model: {MODEL_ID} (REMOTE)")
    print(f"Dataset: {dataset_name}")
    print(f"Classifier: {classifier}")
    print(f"Layers: {layers}")
    print(f"Train: {len(true_train)} true + {len(false_train)} false")
    print(f"Test: {len(true_test)} true + {len(false_test)} false")
    print(f"{'='*60}")
    print("Disk status BEFORE extraction:")
    disk_status()

    probe = LinearProbe(
        model=MODEL_ID,
        layers=layers,
        classifier=classifier,
        remote=True,
        backend="nnsight",
        random_state=RANDOM_STATE,
    )
    probe.fit(true_train, false_train)

    test_prompts = true_test + false_test
    test_labels = [1] * len(true_test) + [0] * len(false_test)
    results = probe.evaluate(test_prompts, test_labels)

    print(f"\nDisk status AFTER extraction:")
    disk_status()

    return {
        "model": MODEL_ID,
        "model_short": MODEL_SHORT,
        "dataset": dataset_name,
        "classifier": classifier,
        "layers": layers,
        "n_train": len(true_train) + len(false_train),
        "n_test": len(true_test) + len(false_test),
        "results": results,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train GoT truth probes — Llama 405B (remote)")
    parser.add_argument(
        "--dataset",
        default=None,
        choices=CURATED_DATASETS,
        help="Single dataset (default: run all)",
    )
    parser.add_argument(
        "--classifier",
        default=None,
        choices=CLASSIFIERS,
        help="Single classifier (default: run all)",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Layer selection: 'all', 'fast_auto', 'middle', or comma-separated indices",
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else CURATED_DATASETS
    classifiers = [args.classifier] if args.classifier else CLASSIFIERS

    # Parse layers
    layers = args.layers
    if "," in layers:
        layers = [int(x) for x in layers.split(",")]

    # Verify NNSIGHT_API_KEY is set
    if not os.environ.get("NNSIGHT_API_KEY"):
        print("ERROR: NNSIGHT_API_KEY environment variable not set.")
        print("Set it with: export NNSIGHT_API_KEY='your-key'")
        return

    print(f"NNSIGHT_API_KEY: {'*' * 8}...set")
    print(f"\nInitial disk status:")
    disk_status()

    all_results = []
    for ds in datasets:
        for clf in classifiers:
            try:
                result = train_and_evaluate(ds, clf, layers)
                all_results.append(result)
                acc = result['results'].get('accuracy', 'N/A')
                auroc = result['results'].get('auroc', 'N/A')
                print(f"\n  -> {ds}/{clf}: Accuracy={acc}, AUROC={auroc}\n")
            except Exception as e:
                print(f"\n  !! ERROR on {ds}/{clf}: {e}\n")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "model": MODEL_ID,
                    "dataset": ds,
                    "classifier": clf,
                    "error": str(e),
                })

            # Save incremental results after each run
            os.makedirs(RESULTS_DIR, exist_ok=True)
            out_path = RESULTS_DIR / f"probes_{MODEL_SHORT}.json"
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    print(f"\nFinal disk status:")
    disk_status()
    print(f"\nCache info:")
    print(cache_info())


if __name__ == "__main__":
    main()
