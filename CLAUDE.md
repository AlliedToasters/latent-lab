# CLAUDE.md — latent-lab

## What This Project Is

`latent-lab` is a HuggingFace organization that publishes linear probes trained with `lmprobe`. The first batch of probes is trained on the **Geometry of Truth** benchmark datasets (Marks & Tegmark, 2023) across small open-weight models.

**The true goal is to stress-test `lmprobe`.** Publishing probes is the vehicle, not the destination. Every rough edge, confusing error, missing feature, or broken workflow you encounter should become a GitHub issue on `AlliedToasters/lmprobe`. The probes we publish are a useful byproduct; the issues we open are the real deliverable.

## Repository: `AlliedToasters/lmprobe`

- **GitHub**: https://github.com/AlliedToasters/lmprobe
- **PyPI**: `pip install lmprobe`
- **Current version**: 0.5.0 (check for updates)
- **Hub integration**: basic plumbing is already implemented (`push_to_hub`, `from_hub`, `ProbeCard`, `evaluate()`)

Install the latest from source to get hub features:
```bash
git clone https://github.com/AlliedToasters/lmprobe.git
cd lmprobe
pip install -e ".[hub,plot]"
```

## Geometry of Truth Datasets

Source: https://github.com/saprmarks/geometry-of-truth

Clone this repo to get the datasets. They live in `datasets/` as CSVs with `statement` and `label` columns (1=true, 0=false).

### Curated datasets (clean, unambiguous — start here):

| Dataset | File | N | Description |
|---------|------|---|-------------|
| cities | `cities.csv` | 1496 | "The city of X is in Y." |
| neg_cities | `neg_cities.csv` | 1496 | "The city of X is not in Y." (negated) |
| sp_en_trans | `sp_en_trans.csv` | 354 | "The Spanish word 'X' means 'Y'." |
| neg_sp_en_trans | `neg_sp_en_trans.csv` | 354 | Negated translations |
| larger_than | `larger_than.csv` | 1980 | "X is larger than Y." |
| smaller_than | `smaller_than.csv` | 1980 | "X is smaller than Y." |

### Uncurated datasets (harder, more diverse — use for generalization tests):

| Dataset | File | N | Description |
|---------|------|---|-------------|
| companies_true_false | `companies_true_false.csv` | 1199 | Company founding claims |
| common_claim_true_false | `common_claim_true_false.csv` | 4450 | Diverse factual claims |
| counterfact_true_false | `counterfact_true_false.csv` | 31964 | From CounterFact benchmark |

### Logical variants (for transfer experiments):

| Dataset | File | N |
|---------|------|---|
| cities_cities_conj | `cities_cities_conj.csv` | 1498 |
| cities_cities_disj | `cities_cities_disj.csv` | 1498 |

## Target Models

Use small open-weight models that run locally on a single consumer GPU (or CPU for the smallest). These are the priority:

| Model | Params | HuggingFace ID |
|-------|--------|----------------|
| Llama 3.2 1B Instruct | 1.2B | `meta-llama/Llama-3.2-1B-Instruct` |
| Llama 3.2 3B Instruct | 3.2B | `meta-llama/Llama-3.2-3B-Instruct` |
| Qwen2.5 0.5B Instruct | 0.5B | `Qwen/Qwen2.5-0.5B-Instruct` |
| Qwen2.5 1.5B Instruct | 1.5B | `Qwen/Qwen2.5-1.5B-Instruct` |
| Qwen2.5 3B Instruct | 3B | `Qwen/Qwen2.5-3B-Instruct` |
| Gemma 2 2B Instruct | 2.6B | `google/gemma-2-2b-it` |

Start with **Qwen2.5-0.5B-Instruct** for fast iteration (it's tiny, fits anywhere), then expand to others. Don't bother with models >4B — they're slow and we're testing lmprobe, not chasing SOTA.

## The Work

### Phase 1: Single-dataset probes (start here)

For each curated dataset × each model, train a truth/falsehood probe:

```python
import pandas as pd
from lmprobe import LinearProbe

df = pd.read_csv("geometry-of-truth/datasets/cities.csv")
true_statements = df[df["label"] == 1]["statement"].tolist()
false_statements = df[df["label"] == 0]["statement"].tolist()

# Hold out 20% for evaluation
from sklearn.model_selection import train_test_split
true_train, true_test = train_test_split(true_statements, test_size=0.2, random_state=42)
false_train, false_test = train_test_split(false_statements, test_size=0.2, random_state=42)

probe = LinearProbe(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    layers="fast_auto",
    classifier="logistic_regression",
    random_state=42,
)
probe.fit(true_train, false_train)

# Evaluate
test_prompts = true_test + false_test
test_labels = [1] * len(true_test) + [0] * len(false_test)
probe.evaluate(test_prompts, test_labels)

# Push
probe.push_to_hub(
    "latent-lab/cities-truth-qwen2.5-0.5b",
    description="Truth/falsehood probe for city-country statements",
    class_labels={0: "false_statement", 1: "true_statement"},
    tags=["safety", "truth", "geometry-of-truth", "cities"],
    include_training_data=True,
    license="mit",
)
```

Vary the classifier types: `logistic_regression`, `mass_mean`, `ridge`, `svm`. The mass_mean classifier is especially relevant — it was introduced in the Geometry of Truth paper itself. Compare them.

Vary layers: try `"fast_auto"`, `"middle"`, specific layers like `[-1]`, `[0]`. See what breaks.

### Phase 2: Cross-dataset generalization

The paper's key finding is that truth probes trained on one dataset can generalize to others. Test this with lmprobe:

```python
# Train on cities
probe = LinearProbe(model="Qwen/Qwen2.5-0.5B-Instruct", layers=16, classifier="mass_mean", random_state=42)
probe.fit(cities_true, cities_false)

# Evaluate on sp_en_trans
probe.score(sp_en_trans_prompts, sp_en_trans_labels)

# Evaluate on neg_cities (should be harder — negation flips the direction)
probe.score(neg_cities_prompts, neg_cities_labels)
```

Build a generalization matrix: train on dataset A, evaluate on datasets B, C, D, ... for each model. This exercises the evaluate/score paths heavily.

### Phase 3: Cross-model comparison

Same dataset, same config, different models. Do smaller models have weaker truth representations? Does the optimal layer differ? This exercises the model-switching workflow and will likely surface caching bugs or device issues.

### Phase 4: Push everything to Hub

Publish the best probes to `latent-lab/` on HuggingFace. Naming convention:
```
latent-lab/{dataset}-truth-{model_short_name}
```
Examples:
- `latent-lab/cities-truth-qwen2.5-0.5b`
- `latent-lab/cities-truth-llama3.2-1b`
- `latent-lab/sp-en-trans-truth-gemma2-2b`

For each probe, make sure to:
1. Call `probe.evaluate()` before pushing
2. Set `class_labels` (always `{0: "false_statement", 1: "true_statement"}`)
3. Set `include_training_data=True`
4. Add tags: `["truth", "geometry-of-truth", "{dataset_name}"]`

After pushing, test the full round-trip:
```python
loaded = LinearProbe.from_hub("latent-lab/cities-truth-qwen2.5-0.5b", load_model=True, trust_classifier=True)
loaded.predict(["The city of Paris is in France."])  # should work end-to-end
```

## How to File Issues

This is the most important part. When something goes wrong — or even just feels clunky — open an issue on https://github.com/AlliedToasters/lmprobe/issues.

### What makes a good issue:

**Bug reports**: exact code that reproduces it, full traceback, Python/lmprobe/torch versions, what you expected vs what happened.

**QoL improvements**: "I had to write 15 lines of boilerplate to do X, here's the API I wish existed." Show the code you wrote and the code you wish you could have written.

**Missing error messages**: "I passed X and got a confusing error. The error should say Y."

**Documentation gaps**: "I couldn't figure out how to do X without reading the source code."

### Issue categories to watch for:

- **Hub integration pain points**: Does `push_to_hub` error on edge cases? Is `from_hub` missing validation? Does `ProbeCard` expose enough info?
- **Data loading friction**: lmprobe expects `list[str]` for prompts. Loading from CSVs and splitting train/test is boilerplate that maybe should be easier.
- **Caching surprises**: Does switching models or layers produce stale cache results? Are cache keys correct?
- **Device/dtype issues**: Does the probe work on CPU? What happens when the model is on GPU but you predict on CPU?
- **Error messages**: Every confusing error is an issue. Every missing error (silent wrong behavior) is a critical issue.
- **Evaluation workflow**: Does `evaluate()` → `push_to_hub()` flow work smoothly? Are metrics correctly rendered on the model card?
- **Baseline comparisons**: Does `BaselineBattery` work out of the box for this use case? Missing baselines?
- **Layer selection**: Does `fast_auto` produce sensible results? Does `"all"` OOM on small machines?

### Labeling convention:

Use these labels when creating issues:
- `bug` — something is broken
- `enhancement` — QoL improvement or missing feature
- `documentation` — docs gap or misleading docs
- `hub` — related to hub integration specifically
- `good first issue` — simple fix, good for contributors

## Project Structure

```
latent-lab/
├── CLAUDE.md              # This file
├── scripts/
│   ├── train_probes.py    # Main training script
│   ├── generalization.py  # Cross-dataset evaluation
│   ├── compare_models.py  # Cross-model comparison
│   └── push_all.py        # Batch push to hub
├── results/
│   ├── metrics/           # JSON files with evaluation results
│   └── figures/           # Layer importance plots, generalization matrices
├── data/                  # Symlink or copy of geometry-of-truth/datasets/
└── notebooks/
    └── exploration.ipynb  # Interactive analysis
```

## Key References

- Marks & Tegmark (2023), "The Geometry of Truth" — https://arxiv.org/abs/2310.06824
- lmprobe README — https://github.com/AlliedToasters/lmprobe
- lmprobe design docs — https://github.com/AlliedToasters/lmprobe/tree/main/docs/design
- HuggingFace Hub integration design — `docs/design/005-hub-integration.md` in lmprobe repo

## Disk Space Management

**lmprobe caches activation tensors aggressively** in `~/.cache/lmprobe/`. A single model × all-layers × large-dataset run can produce tens of GB of cached activations. Left unchecked, this filled a 3.6TB drive to 100%.

Rules:
- **Check `du -sh ~/.cache/lmprobe/` regularly** — especially after running multiple models or `layers="all"`.
- **Prefer `layers="fast_auto"` or specific layer indices** over `layers="all"` to limit cache growth.
- **Clean up after finishing a model's experiments**: `rm -rf ~/.cache/lmprobe/<model_hash>/` for models you're done with.
- **Monitor before long runs**: `df -h /` before kicking off extraction on a new model.
- The HF model cache (`~/.cache/huggingface/`, currently ~156G) is also significant but less volatile.

## Guiding Principles

1. **File issues aggressively.** If you hesitate, file it anyway. We'd rather have 50 issues and close 20 as "working as intended" than miss 30 real problems.
2. **Reproduce before you work around.** When something breaks, write the minimal reproduction case *before* finding a workaround. The reproduction is the issue.
3. **Keep the probes real.** Don't skip evaluation or publish junk. The probes should be genuinely useful, even though the primary goal is stress-testing.
4. **Small models, fast iteration.** Don't wait 20 minutes for a 7B model to extract activations. Use 0.5B-3B models so the feedback loop is tight.
5. **Document everything.** Every workaround, every surprise, every "huh that's weird" moment should be captured somewhere — an issue, a comment in the code, a note in results/.
6. **Manage disk space.** Check cache sizes before and after big runs. Don't let `~/.cache/lmprobe/` silently eat the disk.
