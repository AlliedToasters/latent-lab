"""
Train truth/falsehood probes on Geometry of Truth datasets using SmolLM2-1.7B-Instruct.

GTX 1060 6GB — model is ~3.4GB in fp16, should fit comfortably.
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

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
MODEL_SHORT = "smollm2-1.7b"
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
    print(f"Model: {MODEL_ID}")
    print(f"Dataset: {dataset_name}")
    print(f"Classifier: {classifier}")
    print(f"Layers: {layers}")
    print(f"Train: {len(true_train)} true + {len(false_train)} false")
    print(f"Test: {len(true_test)} true + {len(false_test)} false")
    print(f"{'='*60}\n")

    probe = LinearProbe(
        model=MODEL_ID,
        layers=layers,
        classifier=classifier,
        random_state=RANDOM_STATE,
    )
    probe.fit(true_train, false_train)

    test_prompts = true_test + false_test
    test_labels = [1] * len(true_test) + [0] * len(false_test)
    results = probe.evaluate(test_prompts, test_labels)

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

    parser = argparse.ArgumentParser(description="Train GoT truth probes — SmolLM2 1.7B")
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

    all_results = []
    for ds in datasets:
        for clf in classifiers:
            try:
                result = train_and_evaluate(ds, clf, layers)
                all_results.append(result)
                print(f"\n  -> Accuracy: {result['results'].get('accuracy', 'N/A')}")
                print(f"  -> AUROC: {result['results'].get('auroc', 'N/A')}\n")
            except Exception as e:
                print(f"\n  !! ERROR: {e}\n")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "model": MODEL_ID,
                    "dataset": ds,
                    "classifier": clf,
                    "error": str(e),
                })

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = RESULTS_DIR / f"probes_{MODEL_SHORT}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Cache status
    print(f"\n{'='*60}")
    print("Cache status after run:")
    print(cache_info())
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
