"""
Reproduce Geometry of Truth on LLaMA-2-13B locally (CPU).

This is the paper's primary model — Figure 10 shows clean results.
LLaMA-2-13B: 40 layers, 5120 hidden dim, ~26GB float16.

Paper's Figure 10 reference (train=test diagonal):
  cities     LR: 99  MM: 99
  neg_cities LR: ~90  MM: ~45
  sp_en_trans LR: 84  MM: 95
  larger_than LR: 73  MM: 51
"""

import time
from pathlib import Path

import pandas as pd
from lmprobe import LinearProbe, set_max_threads
from lmprobe.cache import set_cache_limit

set_max_threads(24)
set_cache_limit(None)

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_ID = "meta-llama/Llama-2-13b-hf"

DATASETS = ["cities", "sp_en_trans", "larger_than", "neg_cities"]
# 40 layers — paper used layer 15 for 13B
LAYERS = [5, 10, 12, 15, 18, 20, 25, 30, 35, 39]


def get_prompts(dataset_name):
    df = pd.read_csv(DATA_DIR / f"{dataset_name}.csv")
    true_stmts = df[df["label"] == 1]["statement"].tolist()
    false_stmts = df[df["label"] == 0]["statement"].tolist()
    all_prompts = true_stmts + false_stmts
    all_labels = [1] * len(true_stmts) + [0] * len(false_stmts)
    return true_stmts, false_stmts, all_prompts, all_labels


def main():
    print("=" * 60)
    print(f"Reproducing GoT on {MODEL_ID} (CPU, 24 threads)")
    print("=" * 60)

    for ds in DATASETS:
        true_stmts, false_stmts, all_prompts, all_labels = get_prompts(ds)
        print(f"\n{'='*60}")
        print(f"Dataset: {ds} ({len(all_prompts)} prompts)")
        print(f"{'='*60}")

        for layer in LAYERS:
            for clf in ["logistic_regression", "mass_mean"]:
                kwargs = (
                    {"C": 1.0, "solver": "liblinear", "max_iter": 5000}
                    if clf == "logistic_regression"
                    else {}
                )
                t0 = time.time()
                try:
                    probe = LinearProbe(
                        model=MODEL_ID,
                        layers=[layer],
                        classifier=clf,
                        classifier_kwargs=kwargs,
                        random_state=42,
                    )
                    probe.fit(true_stmts, false_stmts)
                    # Train = test (matching paper's diagonal)
                    ev = probe.evaluate(all_prompts, all_labels)
                    acc = ev.get("accuracy", 0)
                    auroc = ev.get("auroc", 0)
                    elapsed = time.time() - t0
                    print(f"  Layer {layer:3d} {clf:25s}: acc={acc:.4f}  auroc={auroc:.4f}  ({elapsed:.0f}s)")
                except Exception as e:
                    print(f"  Layer {layer:3d} {clf:25s}: FAILED ({e})")

    print("\n\nPaper Figure 10 reference (LLaMA-2-13B, train=test diagonal):")
    print("  cities      LR: 99%  MM: 99%")
    print("  neg_cities  LR: ~90%  MM: ~45%")
    print("  sp_en_trans LR: 84%  MM: 95%")
    print("  larger_than LR: 73%  MM: 51%")


if __name__ == "__main__":
    main()
