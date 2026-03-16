"""
Publish qualifying BitNet b1.58-2B-4T truth probes to latent-lab on HuggingFace Hub.

Exercises lmprobe's full hub integration: evaluate → push_to_hub → from_hub → predict.
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

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"
MODEL_SHORT = "bitnet-2b"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Probes to publish: (dataset, description, tags, exploratory)
PUBLISH_TARGETS = [
    {
        "dataset": "larger_than",
        "description": "Truth probe for 'X is larger than Y' statements. Near-perfect accuracy (99.5%) — structural/relational knowledge survives ternary quantization.",
        "tags": ["truth", "geometry-of-truth", "larger_than", "safety"],
        "exploratory": False,
    },
    {
        "dataset": "smaller_than",
        "description": "Truth probe for 'X is smaller than Y' statements. Near-perfect accuracy (99.0%) — structural/relational knowledge survives ternary quantization.",
        "tags": ["truth", "geometry-of-truth", "smaller_than", "safety"],
        "exploratory": False,
    },
    {
        "dataset": "cities",
        "description": "Truth probe for 'The city of X is in Y' statements. Exploratory — weak signal (81.7%). Semantic/factual knowledge partially degrades under ternary quantization.",
        "tags": ["truth", "geometry-of-truth", "cities", "safety", "exploratory"],
        "exploratory": True,
    },
    {
        "dataset": "neg_cities",
        "description": "Truth probe for 'The city of X is not in Y' (negated) statements. Exploratory — weak signal (80.7%). Semantic/factual knowledge partially degrades under ternary quantization.",
        "tags": ["truth", "geometry-of-truth", "neg_cities", "safety", "exploratory"],
        "exploratory": True,
    },
]


def load_dataset(name: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    true_stmts = df[df["label"] == 1]["statement"].tolist()
    false_stmts = df[df["label"] == 0]["statement"].tolist()
    return true_stmts, false_stmts


def train_evaluate_publish(target: dict) -> dict:
    dataset = target["dataset"]
    true_stmts, false_stmts = load_dataset(dataset)

    true_train, true_test = train_test_split(
        true_stmts, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    false_train, false_test = train_test_split(
        false_stmts, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"\n{'='*60}")
    print(f"Training: {dataset}")
    print(f"Train: {len(true_train)} true + {len(false_train)} false")
    print(f"Test: {len(true_test)} true + {len(false_test)} false")
    print(f"{'='*60}\n")

    probe = LinearProbe(
        model=MODEL_ID,
        layers="all",
        classifier="logistic_regression",
        random_state=RANDOM_STATE,
    )
    probe.fit(true_train, false_train)

    # Evaluate before pushing (required by CLAUDE.md)
    test_prompts = true_test + false_test
    test_labels = [1] * len(true_test) + [0] * len(false_test)
    results = probe.evaluate(test_prompts, test_labels)
    print(f"  Accuracy: {results['accuracy']:.3f}")
    print(f"  AUROC: {results['auroc']:.3f}")

    # Build repo name
    repo_name = f"latent-lab/{dataset.replace('_', '-')}-truth-{MODEL_SHORT}"

    print(f"\n  Pushing to {repo_name}...")
    probe.push_to_hub(
        repo_name,
        description=target["description"],
        class_labels={0: "false_statement", 1: "true_statement"},
        tags=target["tags"],
        include_training_data=True,
        license="mit",
    )
    print(f"  Pushed successfully!")

    # Round-trip test: from_hub → predict
    print(f"\n  Testing round-trip: from_hub → predict...")
    loaded = LinearProbe.from_hub(repo_name, load_model=True, trust_classifier=True)

    # Pick a sample statement to verify prediction works
    sample = true_test[0]
    prediction = loaded.predict([sample])
    print(f"  Sample: {sample!r}")
    print(f"  Prediction: {prediction}")

    return {
        "repo": repo_name,
        "results": results,
        "round_trip": "ok",
    }


def main():
    published = []
    for target in PUBLISH_TARGETS:
        try:
            result = train_evaluate_publish(target)
            published.append(result)
        except Exception as e:
            print(f"\n  !! ERROR publishing {target['dataset']}: {e}")
            import traceback
            traceback.print_exc()
            published.append({
                "dataset": target["dataset"],
                "error": str(e),
            })

    print(f"\n{'='*60}")
    print("Publishing summary:")
    for p in published:
        if "error" in p:
            print(f"  FAILED: {p['dataset']} — {p['error']}")
        else:
            print(f"  OK: {p['repo']} — acc={p['results']['accuracy']:.3f}")
    print(f"{'='*60}")

    # Cache status
    print(f"\nCache: {cache_info()}")


if __name__ == "__main__":
    main()
