"""
Reproduce Geometry of Truth on LLaMA-2-7B and LLaMA-2-13B locally (CPU).

Phase 1: warmup() to cache ALL layers in one forward pass per prompt.
Phase 2: fit probes per layer (instant — cache hits).

Paper reference (train=test diagonal):
  7B  (Fig 11): cities LR:99 MM:99, sp_en_trans LR:84 MM:95, larger_than LR:73 MM:51
  13B (Fig 10): cities LR:99 MM:99, neg_cities LR:~90 MM:~45, sp_en_trans LR:84 MM:95
"""

import gc
import time
from pathlib import Path

import pandas as pd
from lmprobe import LinearProbe, set_max_threads
from lmprobe.cache import set_cache_limit

set_max_threads(24)
set_cache_limit(None)

DATA_DIR = Path(__file__).parent.parent / "data"

MODELS = [
    ("7B", "meta-llama/Llama-2-7b-hf", [5, 10, 15, 20, 25, 30]),
    ("13B", "meta-llama/Llama-2-13b-hf", [5, 10, 12, 15, 18, 20, 25, 30, 35, 39]),
]

DATASETS = ["cities", "neg_cities", "sp_en_trans", "larger_than"]


def get_prompts(dataset_name):
    df = pd.read_csv(DATA_DIR / f"{dataset_name}.csv")
    true_stmts = df[df["label"] == 1]["statement"].tolist()
    false_stmts = df[df["label"] == 0]["statement"].tolist()
    all_prompts = true_stmts + false_stmts
    all_labels = [1] * len(true_stmts) + [0] * len(false_stmts)
    return true_stmts, false_stmts, all_prompts, all_labels


def warmup_model(model_id, label):
    """Cache all layers for all datasets in one pass per dataset."""
    print(f"\n{'#'*60}")
    print(f"# WARMUP: {label} ({model_id})")
    print(f"# Caching ALL layers for all datasets")
    print(f"{'#'*60}")

    for ds in DATASETS:
        true_stmts, false_stmts, all_prompts, all_labels = get_prompts(ds)
        print(f"\n  Warming up {ds} ({len(all_prompts)} prompts)...")

        probe = LinearProbe(
            model=model_id,
            layers="all",
            classifier="mass_mean",
            random_state=42,
        )
        t0 = time.time()
        probe.warmup(all_prompts)
        elapsed = time.time() - t0
        print(f"  -> {ds} cached in {elapsed:.0f}s")
        del probe
        gc.collect()


def eval_model(model_id, label, layers):
    """Fit probes per layer — all activations already cached."""
    print(f"\n{'#'*60}")
    print(f"# EVAL: {label} ({model_id})")
    print(f"# Layers: {layers}")
    print(f"{'#'*60}")

    for ds in DATASETS:
        true_stmts, false_stmts, all_prompts, all_labels = get_prompts(ds)
        print(f"\n  --- {ds} ({len(all_prompts)} prompts) ---")

        for layer in layers:
            for clf in ["logistic_regression", "mass_mean"]:
                kwargs = (
                    {"C": 1.0, "solver": "liblinear", "max_iter": 5000}
                    if clf == "logistic_regression"
                    else {}
                )
                t0 = time.time()
                try:
                    probe = LinearProbe(
                        model=model_id,
                        layers=[layer],
                        classifier=clf,
                        classifier_kwargs=kwargs,
                        random_state=42,
                    )
                    probe.fit(true_stmts, false_stmts)
                    ev = probe.evaluate(all_prompts, all_labels)
                    acc = ev.get("accuracy", 0)
                    auroc = ev.get("auroc", 0)
                    elapsed = time.time() - t0
                    print(f"    Layer {layer:3d} {clf:25s}: acc={acc:.4f}  auroc={auroc:.4f}  ({elapsed:.0f}s)")
                except Exception as e:
                    print(f"    Layer {layer:3d} {clf:25s}: FAILED ({e})")


def main():
    print("=" * 60)
    print("Reproducing Geometry of Truth (Marks & Tegmark 2023)")
    print("LLaMA-2-7B and LLaMA-2-13B on CPU")
    print("=" * 60)

    for label, model_id, layers in MODELS:
        # Phase 1: warmup (one forward pass per prompt, caches all layers)
        warmup_model(model_id, label)

        # Free the model from memory before fitting
        gc.collect()

        # Phase 2: fit probes (cache hits, fast)
        eval_model(model_id, label, layers)

        # Free memory before loading next model
        gc.collect()

    print("\n\nPaper reference (train=test diagonal):")
    print("  7B  cities: LR 99, MM 99 | sp_en_trans: LR 84, MM 95 | larger_than: LR 73, MM 51")
    print("  13B cities: LR 99, MM 99 | neg_cities: LR ~90, MM ~45 | sp_en_trans: LR 84, MM 95")


if __name__ == "__main__":
    main()
