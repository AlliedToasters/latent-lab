"""
Train truth/falsehood probes on Geometry of Truth datasets using lmprobe.

Reproduces results from ../bitnet_got/ but through the lmprobe library
to stress-test it. Starts with BitNet b1.58-2B-4T.
"""

import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from lmprobe import LinearProbe, enable_cache_logging, set_max_threads
from lmprobe.cache import set_cache_limit, set_cache_dtype, cache_info

# Shared machine — limit CPU threads
set_max_threads(4)

# Cache: float16 storage, 50GB cap
set_cache_dtype("float16")
set_cache_limit(50)
enable_cache_logging()

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "metrics"

# Datasets to probe (curated ones first)
CURATED_DATASETS = [
    "cities",
    "neg_cities",
    "sp_en_trans",
    "neg_sp_en_trans",
    "larger_than",
    "smaller_than",
]

# Models to test
MODELS = {
    "bitnet-2b": "microsoft/bitnet-b1.58-2B-4T",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
}

# Classifiers to compare
CLASSIFIERS = ["logistic_regression", "mass_mean"]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset(name: str) -> tuple[list[str], list[str]]:
    """Load a GoT dataset, returning (true_statements, false_statements)."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    true_stmts = df[df["label"] == 1]["statement"].tolist()
    false_stmts = df[df["label"] == 0]["statement"].tolist()
    return true_stmts, false_stmts


def train_and_evaluate(
    model_id: str,
    dataset_name: str,
    classifier: str,
    layers: str = "all",
    device: str = "auto",
) -> dict:
    """Train a probe and return evaluation metrics."""
    true_stmts, false_stmts = load_dataset(dataset_name)

    # Train/test split
    true_train, true_test = train_test_split(
        true_stmts, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    false_train, false_test = train_test_split(
        false_stmts, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"\n{'='*60}")
    print(f"Model: {model_id}")
    print(f"Dataset: {dataset_name}")
    print(f"Classifier: {classifier}")
    print(f"Layers: {layers}")
    print(f"Train: {len(true_train)} true + {len(false_train)} false")
    print(f"Test: {len(true_test)} true + {len(false_test)} false")
    print(f"{'='*60}\n")

    probe = LinearProbe(
        model=model_id,
        layers=layers,
        classifier=classifier,
        device=device,
        random_state=RANDOM_STATE,
    )
    probe.fit(true_train, false_train)

    # Evaluate
    test_prompts = true_test + false_test
    test_labels = [1] * len(true_test) + [0] * len(false_test)
    results = probe.evaluate(test_prompts, test_labels)

    return {
        "model": model_id,
        "dataset": dataset_name,
        "classifier": classifier,
        "layers": layers,
        "n_train": len(true_train) + len(false_train),
        "n_test": len(true_test) + len(false_test),
        "results": results,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train GoT truth probes with lmprobe")
    parser.add_argument(
        "--model",
        default="bitnet-2b",
        choices=list(MODELS.keys()),
        help="Model short name",
    )
    parser.add_argument(
        "--dataset",
        default="larger_than",
        choices=CURATED_DATASETS,
        help="Dataset to probe",
    )
    parser.add_argument(
        "--classifier",
        default="logistic_regression",
        choices=CLASSIFIERS,
        help="Classifier type",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Layer selection: 'all', 'fast_auto', 'middle', or comma-separated indices",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run all curated datasets for the given model",
    )
    parser.add_argument(
        "--all-classifiers",
        action="store_true",
        help="Run all classifiers for the given dataset(s)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: 'auto', 'cpu', 'cuda'",
    )
    args = parser.parse_args()

    model_id = MODELS[args.model]
    datasets = CURATED_DATASETS if args.all_datasets else [args.dataset]
    classifiers = CLASSIFIERS if args.all_classifiers else [args.classifier]

    # Parse layers
    layers = args.layers
    if "," in layers:
        layers = [int(x) for x in layers.split(",")]

    all_results = []
    for ds in datasets:
        for clf in classifiers:
            try:
                result = train_and_evaluate(model_id, ds, clf, layers, device=args.device)
                all_results.append(result)
                print(f"\n  -> Results: {result['results']}\n")
            except Exception as e:
                print(f"\n  !! ERROR: {e}\n")
                all_results.append(
                    {
                        "model": model_id,
                        "dataset": ds,
                        "classifier": clf,
                        "error": str(e),
                    }
                )

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = RESULTS_DIR / f"probes_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Report cache state
    print(f"\n{'='*60}")
    print("Cache status after run:")
    print(cache_info())
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
