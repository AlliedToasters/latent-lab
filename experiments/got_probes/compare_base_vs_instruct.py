"""
Compare truth probe accuracy: Llama 3.1 70B Base vs Instruct.

Tests the hypothesis that instruct fine-tuning muddles truth representations,
which would explain why our 405B-Instruct probes plateau at ~72% on cities
while Marks & Tegmark (2023) got better results on base models.

Both models' activations are fully cached (601GB total). No GPU needed.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lmprobe import LinearProbe, set_max_threads
from lmprobe.cache import set_cache_limit

set_max_threads(8)
set_cache_limit(None)

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "metrics"
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

# 70B has 80 layers (0-79), 8192 hidden dim
# Sweep a range of layers to find the best for each model
SWEEP_LAYERS = [10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 79]


def get_prompts_and_labels(dataset_name):
    """Load dataset, split into train/test prompts and labels."""
    df = pd.read_csv(DATA_DIR / f"{dataset_name}.csv")
    true_stmts = df[df["label"] == 1]["statement"].tolist()
    false_stmts = df[df["label"] == 0]["statement"].tolist()

    true_train, true_test = train_test_split(
        true_stmts, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    false_train, false_test = train_test_split(
        false_stmts, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return {
        "true_train": true_train,
        "true_test": true_test,
        "false_train": false_train,
        "false_test": false_test,
        "test_prompts": true_test + false_test,
        "test_labels": [1] * len(true_test) + [0] * len(false_test),
    }


def run_layer_sweep(model_key, model_id, dataset="cities"):
    """Sweep layers on cities to find best layer for each model."""
    print(f"\n{'#'*60}")
    print(f"# Layer sweep: {model_key} ({model_id})")
    print(f"# Dataset: {dataset}, Layers: {SWEEP_LAYERS}")
    print(f"{'#'*60}")

    data = get_prompts_and_labels(dataset)
    results = {}

    for layer in SWEEP_LAYERS:
        t0 = time.time()
        probe = LinearProbe(
            model=model_id,
            layers=[layer],
            classifier="logistic_regression",
            preprocessing="standard+pca",
            pca_components=200,
            classifier_kwargs={"C": 0.5, "solver": "liblinear", "max_iter": 5000},
            random_state=RANDOM_STATE,
        )
        probe.fit(data["true_train"], data["false_train"])
        eval_result = probe.evaluate(data["test_prompts"], data["test_labels"])

        acc = eval_result.get("accuracy", 0)
        auroc = eval_result.get("auroc", 0)
        elapsed = time.time() - t0

        results[layer] = {"accuracy": acc, "auroc": auroc, "time_s": round(elapsed, 1)}
        print(f"  Layer {layer:3d}: acc={acc:.4f}  auroc={auroc:.4f}  ({elapsed:.1f}s)")

    # Find best layer
    best_layer = max(results, key=lambda l: results[l]["accuracy"])
    best_auroc_layer = max(results, key=lambda l: results[l]["auroc"])
    print(f"\n  Best accuracy: layer {best_layer} ({results[best_layer]['accuracy']:.4f})")
    print(f"  Best AUROC:    layer {best_auroc_layer} ({results[best_auroc_layer]['auroc']:.4f})")

    return results, best_layer


def run_full_eval(model_key, model_id, layer):
    """Run full evaluation on all datasets at the best layer."""
    print(f"\n{'#'*60}")
    print(f"# Full eval: {model_key} @ layer {layer}")
    print(f"{'#'*60}")

    all_results = {}
    for ds in CURATED_DATASETS:
        data = get_prompts_and_labels(ds)
        t0 = time.time()

        probe = LinearProbe(
            model=model_id,
            layers=[layer],
            classifier="logistic_regression",
            preprocessing="standard+pca",
            pca_components=200,
            classifier_kwargs={"C": 0.5, "solver": "liblinear", "max_iter": 5000},
            random_state=RANDOM_STATE,
        )
        probe.fit(data["true_train"], data["false_train"])
        eval_result = probe.evaluate(data["test_prompts"], data["test_labels"])

        acc = eval_result.get("accuracy", 0)
        auroc = eval_result.get("auroc", 0)
        elapsed = time.time() - t0

        all_results[ds] = {"accuracy": acc, "auroc": auroc, "time_s": round(elapsed, 1)}
        print(f"  {ds:20s}: acc={acc:.4f}  auroc={auroc:.4f}  ({elapsed:.1f}s)")

    # Aggregate
    accuracies = [r["accuracy"] for r in all_results.values()]
    aurocs = [r["auroc"] for r in all_results.values()]
    all_results["_aggregate"] = {
        "mean_accuracy": float(np.mean(accuracies)),
        "mean_auroc": float(np.mean(aurocs)),
        "cities_accuracy": all_results["cities"]["accuracy"],
    }

    return all_results


def main():
    print("=" * 60)
    print("Llama 3.1 70B: Base vs Instruct Truth Probes")
    print("=" * 60)

    all_data = {}

    # Phase 1: Layer sweep on cities for both models
    for model_key, model_id in MODELS.items():
        sweep_results, best_layer = run_layer_sweep(model_key, model_id)
        all_data[model_key] = {
            "model_id": model_id,
            "sweep_results": {str(k): v for k, v in sweep_results.items()},
            "best_layer": best_layer,
        }

    # Phase 2: Full eval at best layer for each model
    for model_key, model_id in MODELS.items():
        best_layer = all_data[model_key]["best_layer"]
        full_results = run_full_eval(model_key, model_id, best_layer)
        all_data[model_key]["full_results"] = full_results

    # Phase 3: Head-to-head comparison
    print(f"\n{'='*60}")
    print(f"HEAD-TO-HEAD COMPARISON")
    print(f"{'='*60}")

    base_best = all_data["base"]["best_layer"]
    inst_best = all_data["instruct"]["best_layer"]
    print(f"\n  Best layer — Base: {base_best}, Instruct: {inst_best}")

    print(f"\n  {'Dataset':<20s} {'Base':>10s} {'Instruct':>10s} {'Delta':>10s}")
    print(f"  {'-'*50}")

    for ds in CURATED_DATASETS:
        base_acc = all_data["base"]["full_results"][ds]["accuracy"]
        inst_acc = all_data["instruct"]["full_results"][ds]["accuracy"]
        delta = base_acc - inst_acc
        marker = " ***" if abs(delta) > 0.05 else ""
        print(f"  {ds:<20s} {base_acc:>10.4f} {inst_acc:>10.4f} {delta:>+10.4f}{marker}")

    base_mean = all_data["base"]["full_results"]["_aggregate"]["mean_accuracy"]
    inst_mean = all_data["instruct"]["full_results"]["_aggregate"]["mean_accuracy"]
    print(f"  {'-'*50}")
    print(f"  {'MEAN':<20s} {base_mean:>10.4f} {inst_mean:>10.4f} {base_mean - inst_mean:>+10.4f}")

    base_auroc = all_data["base"]["full_results"]["_aggregate"]["mean_auroc"]
    inst_auroc = all_data["instruct"]["full_results"]["_aggregate"]["mean_auroc"]
    print(f"  {'MEAN AUROC':<20s} {base_auroc:>10.4f} {inst_auroc:>10.4f} {base_auroc - inst_auroc:>+10.4f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "base_vs_instruct_70b.json"
    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
