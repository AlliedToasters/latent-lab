"""
Single-layer probe sweep for Llama 3.1 405B.

All activations are already cached (368GB, all 126 layers).
We can't load all layers at once (OOM), so we sweep individual layers
to find the best ones, then hone in.

Strategy:
  1. Coarse sweep: every 10th layer on cities dataset
  2. Fine sweep: dense search around the best layers
  3. Full run: best layers × all datasets × both classifiers
"""

import gc
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from lmprobe import LinearProbe, set_max_threads
from lmprobe.cache import set_cache_dtype

set_max_threads(8)
set_cache_dtype("float16")

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "metrics"

MODEL_ID = "meta-llama/Llama-3.1-405B-Instruct"
MODEL_SHORT = "llama3.1-405b"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_LAYERS = 126  # layers 0-125

CURATED_DATASETS = [
    "cities",
    "neg_cities",
    "sp_en_trans",
    "neg_sp_en_trans",
    "larger_than",
    "smaller_than",
]

CLASSIFIERS = ["logistic_regression", "mass_mean"]


def load_dataset(name: str):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    true_stmts = df[df["label"] == 1]["statement"].tolist()
    false_stmts = df[df["label"] == 0]["statement"].tolist()
    return true_stmts, false_stmts


def split_dataset(name: str):
    true_stmts, false_stmts = load_dataset(name)
    true_train, true_test = train_test_split(
        true_stmts, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    false_train, false_test = train_test_split(
        false_stmts, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return true_train, true_test, false_train, false_test


def train_single_layer(dataset_name, layer, classifier):
    """Train and evaluate a probe on a single layer. Returns result dict."""
    true_train, true_test, false_train, false_test = split_dataset(dataset_name)

    probe = LinearProbe(
        model=MODEL_ID,
        layers=[layer],
        classifier=classifier,
        remote=True,
        backend="nnsight",
        random_state=RANDOM_STATE,
    )
    probe.fit(true_train, false_train)

    test_prompts = true_test + false_test
    test_labels = [1] * len(true_test) + [0] * len(false_test)
    results = probe.evaluate(test_prompts, test_labels)

    del probe
    gc.collect()

    return {
        "model": MODEL_ID,
        "model_short": MODEL_SHORT,
        "dataset": dataset_name,
        "classifier": classifier,
        "layer": layer,
        "n_train": len(true_train) + len(false_train),
        "n_test": len(true_test) + len(false_test),
        "results": results,
    }


def save_results(all_results, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = RESULTS_DIR / filename
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    return out_path


def coarse_sweep(dataset="cities", classifiers=None, step=10):
    """Sweep every `step`-th layer on a single dataset."""
    if classifiers is None:
        classifiers = CLASSIFIERS

    layers = list(range(0, N_LAYERS, step))
    if (N_LAYERS - 1) not in layers:
        layers.append(N_LAYERS - 1)  # always include last layer

    print(f"Coarse sweep: {len(layers)} layers × {len(classifiers)} classifiers on {dataset}")
    print(f"Layers: {layers}")

    all_results = []
    for layer in layers:
        for clf in classifiers:
            try:
                result = train_single_layer(dataset, layer, clf)
                all_results.append(result)
                acc = result["results"].get("accuracy", "N/A")
                auroc = result["results"].get("auroc", "N/A")
                print(f"  Layer {layer:3d} / {clf:20s}: acc={acc:.4f}  auroc={auroc:.4f}")
            except Exception as e:
                print(f"  Layer {layer:3d} / {clf:20s}: ERROR — {e}")
                all_results.append({
                    "model": MODEL_ID,
                    "dataset": dataset,
                    "classifier": clf,
                    "layer": layer,
                    "error": str(e),
                })

            save_results(all_results, f"layer_sweep_{MODEL_SHORT}_coarse.json")

    return all_results


def fine_sweep(center_layers, dataset="cities", classifiers=None, radius=5):
    """Dense sweep around the best layers from coarse sweep."""
    if classifiers is None:
        classifiers = CLASSIFIERS

    layers = set()
    for center in center_layers:
        for offset in range(-radius, radius + 1):
            layer = center + offset
            if 0 <= layer < N_LAYERS:
                layers.add(layer)
    layers = sorted(layers)

    print(f"\nFine sweep: {len(layers)} layers × {len(classifiers)} classifiers on {dataset}")
    print(f"Layers: {layers}")

    all_results = []
    for layer in layers:
        for clf in classifiers:
            try:
                result = train_single_layer(dataset, layer, clf)
                all_results.append(result)
                acc = result["results"].get("accuracy", "N/A")
                auroc = result["results"].get("auroc", "N/A")
                print(f"  Layer {layer:3d} / {clf:20s}: acc={acc:.4f}  auroc={auroc:.4f}")
            except Exception as e:
                print(f"  Layer {layer:3d} / {clf:20s}: ERROR — {e}")
                all_results.append({
                    "model": MODEL_ID,
                    "dataset": dataset,
                    "classifier": clf,
                    "layer": layer,
                    "error": str(e),
                })

            save_results(all_results, f"layer_sweep_{MODEL_SHORT}_fine.json")

    return all_results


def full_run(layers, datasets=None, classifiers=None):
    """Run best layers across all datasets and classifiers."""
    if datasets is None:
        datasets = CURATED_DATASETS
    if classifiers is None:
        classifiers = CLASSIFIERS

    print(f"\nFull run: {len(layers)} layers × {len(datasets)} datasets × {len(classifiers)} classifiers")
    print(f"Layers: {layers}")

    all_results = []
    for ds in datasets:
        for layer in layers:
            for clf in classifiers:
                try:
                    result = train_single_layer(ds, layer, clf)
                    all_results.append(result)
                    acc = result["results"].get("accuracy", "N/A")
                    auroc = result["results"].get("auroc", "N/A")
                    print(f"  {ds:20s} L{layer:3d} / {clf:20s}: acc={acc:.4f}  auroc={auroc:.4f}")
                except Exception as e:
                    print(f"  {ds:20s} L{layer:3d} / {clf:20s}: ERROR — {e}")
                    all_results.append({
                        "model": MODEL_ID,
                        "dataset": ds,
                        "classifier": clf,
                        "layer": layer,
                        "error": str(e),
                    })

                save_results(all_results, f"probes_{MODEL_SHORT}.json")

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Layer sweep for 405B probes")
    parser.add_argument(
        "mode",
        choices=["coarse", "fine", "full"],
        help="Sweep mode: coarse (every 10th layer), fine (dense around best), full (best layers × all datasets)",
    )
    parser.add_argument(
        "--dataset", default="cities", choices=CURATED_DATASETS,
        help="Dataset for coarse/fine sweep (default: cities)",
    )
    parser.add_argument(
        "--layers", default=None,
        help="Comma-separated layer indices (for fine: centers; for full: layers to use)",
    )
    parser.add_argument(
        "--step", type=int, default=10,
        help="Step size for coarse sweep (default: 10)",
    )
    parser.add_argument(
        "--radius", type=int, default=5,
        help="Radius around center layers for fine sweep (default: 5)",
    )
    parser.add_argument(
        "--classifier", default=None, choices=CLASSIFIERS,
        help="Single classifier (default: run both)",
    )
    args = parser.parse_args()

    classifiers = [args.classifier] if args.classifier else None

    if args.mode == "coarse":
        results = coarse_sweep(args.dataset, classifiers, args.step)

        # Print summary sorted by accuracy
        print(f"\n{'='*70}")
        print("COARSE SWEEP SUMMARY (sorted by accuracy)")
        print(f"{'='*70}")
        valid = [r for r in results if "error" not in r]
        valid.sort(key=lambda r: r["results"].get("accuracy", 0), reverse=True)
        for r in valid:
            acc = r["results"].get("accuracy", 0)
            auroc = r["results"].get("auroc", 0)
            print(f"  Layer {r['layer']:3d} / {r['classifier']:20s}: acc={acc:.4f}  auroc={auroc:.4f}")

        # Suggest fine sweep centers
        if valid:
            top_layers = sorted(set(r["layer"] for r in valid[:6]))
            print(f"\nSuggested fine sweep: --layers {','.join(map(str, top_layers))}")

    elif args.mode == "fine":
        if not args.layers:
            print("ERROR: --layers required for fine sweep (comma-separated center layers)")
            return
        centers = [int(x) for x in args.layers.split(",")]
        fine_sweep(centers, args.dataset, classifiers, args.radius)

    elif args.mode == "full":
        if not args.layers:
            print("ERROR: --layers required for full run")
            return
        layers = [int(x) for x in args.layers.split(",")]
        full_run(layers, classifiers=classifiers)


if __name__ == "__main__":
    main()
