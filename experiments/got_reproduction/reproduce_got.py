"""
Reproduce Geometry of Truth (Marks & Tegmark, 2023) methodology.

The paper used:
- Base models (LLaMA-13B originally, we use Llama 3.1 70B base)
- mass_mean classifier (class centroids → separating direction)
- No PCA, no preprocessing
- Middle layers

Compare mass_mean vs logistic_regression, with and without preprocessing,
on the 70B base model to see how much our pipeline choices matter.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lmprobe import LinearProbe, set_max_threads
from lmprobe.cache import set_cache_limit

set_max_threads(8)
set_cache_limit(None)

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "metrics"
RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_BASE = "meta-llama/Llama-3.1-70B"
MODEL_INSTRUCT = "meta-llama/Llama-3.1-70B-Instruct"

CURATED_DATASETS = [
    "cities",
    "neg_cities",
    "sp_en_trans",
    "neg_sp_en_trans",
    "larger_than",
    "smaller_than",
]

# 70B: 80 layers, 8192 hidden dim
SWEEP_LAYERS = list(range(10, 80, 5))  # Every 5th layer from 10-79


def get_prompts_and_labels(dataset_name):
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
        "true_train": true_train, "true_test": true_test,
        "false_train": false_train, "false_test": false_test,
        "test_prompts": true_test + false_test,
        "test_labels": [1] * len(true_test) + [0] * len(false_test),
    }


def sweep_config(model_id, classifier, preprocessing, pca_components,
                 classifier_kwargs, label, dataset="cities"):
    """Sweep layers for a given config."""
    print(f"\n  --- {label} ---")
    data = get_prompts_and_labels(dataset)
    results = {}

    for layer in SWEEP_LAYERS:
        t0 = time.time()
        try:
            probe = LinearProbe(
                model=model_id,
                layers=[layer],
                classifier=classifier,
                preprocessing=preprocessing,
                pca_components=pca_components,
                classifier_kwargs=classifier_kwargs or {},
                random_state=RANDOM_STATE,
            )
            probe.fit(data["true_train"], data["false_train"])
            ev = probe.evaluate(data["test_prompts"], data["test_labels"])
            acc = ev.get("accuracy", 0)
            auroc = ev.get("auroc", 0)
        except Exception as e:
            acc, auroc = 0, 0
            print(f"    Layer {layer}: FAILED ({e})")
            continue

        elapsed = time.time() - t0
        results[layer] = {"accuracy": acc, "auroc": auroc}
        print(f"    Layer {layer:3d}: acc={acc:.4f}  auroc={auroc:.4f}  ({elapsed:.1f}s)")

    if results:
        best_layer = max(results, key=lambda l: results[l]["accuracy"])
        print(f"    -> Best: layer {best_layer} acc={results[best_layer]['accuracy']:.4f}")
    return results


def full_eval(model_id, layer, classifier, preprocessing, pca_components,
              classifier_kwargs, label):
    """Evaluate on all datasets at a given layer."""
    print(f"\n  --- {label} @ layer {layer} ---")
    all_results = {}

    for ds in CURATED_DATASETS:
        data = get_prompts_and_labels(ds)
        try:
            probe = LinearProbe(
                model=model_id,
                layers=[layer],
                classifier=classifier,
                preprocessing=preprocessing,
                pca_components=pca_components,
                classifier_kwargs=classifier_kwargs or {},
                random_state=RANDOM_STATE,
            )
            probe.fit(data["true_train"], data["false_train"])
            ev = probe.evaluate(data["test_prompts"], data["test_labels"])
            acc = ev.get("accuracy", 0)
            auroc = ev.get("auroc", 0)
        except Exception as e:
            acc, auroc = 0, 0
            print(f"    {ds}: FAILED ({e})")
            continue

        all_results[ds] = {"accuracy": acc, "auroc": auroc}
        print(f"    {ds:20s}: acc={acc:.4f}  auroc={auroc:.4f}")

    accuracies = [r["accuracy"] for r in all_results.values()]
    print(f"    MEAN: {np.mean(accuracies):.4f}")
    return all_results


def main():
    configs = [
        # (classifier, preprocessing, pca_components, classifier_kwargs, label)
        ("mass_mean", None, None, None, "mass_mean (paper method)"),
        ("mass_mean", "standard", None, None, "mass_mean + standard"),
        ("mass_mean", "standard+pca", 200, None, "mass_mean + std+pca(200)"),
        ("logistic_regression", None, None,
         {"C": 0.5, "solver": "liblinear", "max_iter": 5000},
         "logreg (no preproc)"),
        ("logistic_regression", "standard+pca", 200,
         {"C": 0.5, "solver": "liblinear", "max_iter": 5000},
         "logreg + std+pca(200)"),
        ("lda", "standard+pca", 200, None, "LDA + std+pca(200)"),
    ]

    all_data = {"base": {}, "instruct": {}}

    for model_key, model_id in [("base", MODEL_BASE), ("instruct", MODEL_INSTRUCT)]:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_key} ({model_id})")
        print(f"{'='*60}")

        # Sweep each config
        for clf, preproc, pca, kwargs, label in configs:
            key = label.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
            sweep = sweep_config(model_id, clf, preproc, pca, kwargs, label)
            all_data[model_key][key] = {
                "config": {"classifier": clf, "preprocessing": preproc,
                           "pca_components": pca, "classifier_kwargs": kwargs},
                "sweep": {str(k): v for k, v in sweep.items()},
            }
            if sweep:
                best_layer = max(sweep, key=lambda l: sweep[l]["accuracy"])
                all_data[model_key][key]["best_layer"] = best_layer
                all_data[model_key][key]["best_accuracy"] = sweep[best_layer]["accuracy"]

    # Full eval: paper method (mass_mean, no preproc) at best layer for each model
    print(f"\n{'='*60}")
    print(f"FULL EVAL: Paper method (mass_mean, no preprocessing)")
    print(f"{'='*60}")

    mm_key = "mass_mean_paper_method"
    for model_key, model_id in [("base", MODEL_BASE), ("instruct", MODEL_INSTRUCT)]:
        best_layer = all_data[model_key][mm_key]["best_layer"]
        full = full_eval(model_id, best_layer, "mass_mean", None, None, None,
                         f"{model_key} mass_mean")
        all_data[model_key][mm_key]["full_eval"] = full

    # Summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY: Best cities accuracy per config")
    print(f"{'='*60}")
    print(f"\n  {'Config':<35s} {'Base':>8s} {'Instruct':>10s} {'Delta':>8s}")
    print(f"  {'-'*63}")

    for clf, preproc, pca, kwargs, label in configs:
        key = label.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
        base_acc = all_data["base"].get(key, {}).get("best_accuracy", 0)
        inst_acc = all_data["instruct"].get(key, {}).get("best_accuracy", 0)
        delta = base_acc - inst_acc
        print(f"  {label:<35s} {base_acc:>8.4f} {inst_acc:>10.4f} {delta:>+8.4f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "reproduce_got_70b.json"
    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
