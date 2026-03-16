"""
Download LODO benchmark datasets and save as standardized Parquet files.

Each output file has columns: prompt (str), malicious (bool), dataset_id (str).
Some datasets include additional metadata columns where available.

Two datasets (BIPIA, InjecAgent) require manual git clones — this script
handles the 16 HuggingFace-hosted datasets and prints instructions for
the manual ones.

Source: https://github.com/maxf-zn/prompt-mining
Paper: "When Benchmarks Lie" (arXiv:2602.14161)
"""

import subprocess
from pathlib import Path

import pandas as pd
from datasets import load_dataset

LODO_DIR = Path(__file__).parent.parent / "data" / "lodo"
LODO_DIR.mkdir(parents=True, exist_ok=True)


def save(df: pd.DataFrame, name: str):
    """Save a DataFrame as parquet with standard columns."""
    assert "prompt" in df.columns, f"Missing 'prompt' column in {name}"
    assert "malicious" in df.columns, f"Missing 'malicious' column in {name}"
    assert "dataset_id" in df.columns, f"Missing 'dataset_id' column in {name}"
    out = LODO_DIR / f"{name}.parquet"
    df.to_parquet(out, index=False)
    n_mal = df["malicious"].sum()
    print(f"  {name}: {len(df)} rows ({n_mal} malicious, {len(df)-n_mal} benign) -> {out.name}")


def download_advbench():
    ds = load_dataset("walledai/AdvBench", split="train")
    df = pd.DataFrame({"prompt": ds["prompt"], "malicious": True, "dataset_id": "advbench"})
    save(df, "advbench")


def download_harmbench():
    # Has configs: standard, contextual, copyright — load all
    rows = []
    for config in ["standard", "contextual", "copyright"]:
        try:
            ds = load_dataset("walledai/HarmBench", config, split="train")
            for ex in ds:
                rows.append({"prompt": ex["prompt"], "malicious": True, "dataset_id": "harmbench"})
        except Exception as e:
            print(f"  WARNING: HarmBench config '{config}' failed: {e}")
    save(pd.DataFrame(rows), "harmbench")


def download_wildjailbreak():
    # LODO uses the train config (262K rows), not eval (2.2K).
    # Columns: vanilla, adversarial, completion, data_type
    # Prefer adversarial text, fall back to vanilla. Ignore completion.
    # Malicious = "harmful" in data_type.
    ds = load_dataset("allenai/wildjailbreak", "train", split="train", streaming=True)
    rows = []
    for ex in ds:
        prompt = ex.get("adversarial") or ex.get("vanilla", "")
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            continue
        malicious = "harmful" in ex.get("data_type", "")
        rows.append({"prompt": prompt.strip(), "malicious": malicious, "dataset_id": "wildjailbreak"})
    save(pd.DataFrame(rows), "wildjailbreak")


def download_yanismiraoui():
    ds = load_dataset("yanismiraoui/prompt_injections", split="train")
    df = pd.DataFrame(ds)
    # Single column 'prompt_injections' containing the text; all are injections
    result = pd.DataFrame({
        "prompt": df["prompt_injections"],
        "malicious": True,
        "dataset_id": "yanismiraoui",
    })
    save(result, "yanismiraoui")


def download_llmail():
    # Actual repo is microsoft/llmail-inject-challenge with splits Phase1/Phase2
    # All prompts are injection attempts (malicious) — it's a red-team challenge dataset
    import json
    rows = []
    for split in ["Phase1", "Phase2"]:
        ds = load_dataset("microsoft/llmail-inject-challenge", split=split)
        for ex in ds:
            body = ex.get("body", "")
            subject = ex.get("subject", "")
            prompt = f"Subject: {subject}\n\n{body}".strip() if subject else body
            if prompt:
                rows.append({"prompt": prompt, "malicious": True, "dataset_id": "llmail"})
    # Cap at 10k to keep manageable
    df = pd.DataFrame(rows)
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42).reset_index(drop=True)
    save(df, "llmail")


def download_mosscap():
    ds = load_dataset("Lakera/mosscap_prompt_injection", split="train")
    df = pd.DataFrame(ds)
    # All prompts are password extraction attacks (malicious)
    result = pd.DataFrame({
        "prompt": df["prompt"],
        "malicious": True,
        "dataset_id": "mosscap",
    })
    save(result, "mosscap")


def download_gandalf():
    ds = load_dataset("Lakera/gandalf_summarization", split="train")
    df = pd.DataFrame(ds)
    prompt_col = next(c for c in ["prompt", "text", "input"] if c in df.columns)
    result = pd.DataFrame({
        "prompt": df[prompt_col],
        "malicious": True,
        "dataset_id": "gandalf",
    })
    save(result, "gandalf")


def download_jayavibhav():
    ds = load_dataset("jayavibhav/prompt-injection", split="train")
    df = pd.DataFrame(ds)
    result = pd.DataFrame({
        "prompt": df["text"],
        "malicious": df["label"] == 1,
        "dataset_id": "jayavibhav",
    })
    save(result, "jayavibhav")


def download_qualifire():
    # qualifire/safety-benchmark — test split only
    # Columns: text, Sexually Explicit Information, Harassment, Hate Speech, Dangerous Content, Safe
    # Malicious if Safe == "0"
    ds = load_dataset("qualifire/safety-benchmark", split="test")
    df = pd.DataFrame(ds)
    result = pd.DataFrame({
        "prompt": df["text"],
        "malicious": df["Safe"].astype(str) == "0",
        "dataset_id": "qualifire",
    })
    save(result, "qualifire")


def download_safeguard():
    ds = load_dataset("xTRam1/safe-guard-prompt-injection", split="train")
    df = pd.DataFrame(ds)
    # Columns: text, label (1=malicious, 0=benign)
    result = pd.DataFrame({
        "prompt": df["text"],
        "malicious": df["label"] == 1,
        "dataset_id": "safeguard",
    })
    save(result, "safeguard")


def download_deepset():
    ds = load_dataset("deepset/prompt-injections", split="train")
    df = pd.DataFrame(ds)
    result = pd.DataFrame({
        "prompt": df["text"],
        "malicious": df["label"] == 1,
        "dataset_id": "deepset",
    })
    save(result, "deepset")


def download_enron():
    # The HF datasets lib has a schema mismatch — read parquet shards directly
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    shards = [
        "data/train-00000-of-00001-c67cd1a5d0208775.parquet",  # 36K rows
        "data/test-00000-of-00001-32266e0231f6eae6.parquet",   # 4K rows
    ]
    tables = []
    for shard in shards:
        path = hf_hub_download(
            repo_id="amanneo/enron-mail-corpus-mini",
            filename=shard,
            repo_type="dataset",
        )
        tables.append(pq.read_table(path, columns=["text"]))
    import pyarrow as pa
    table = pa.concat_tables(tables)
    df = table.to_pandas()
    result = pd.DataFrame({
        "prompt": df["text"],
        "malicious": False,
        "dataset_id": "enron",
    })
    # Cap at 10k to match LODO paper
    if len(result) > 10000:
        result = result.sample(n=10000, random_state=42).reset_index(drop=True)
    save(result, "enron")


def download_openorca():
    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    rows = []
    for i, ex in enumerate(ds):
        if i >= 10000:
            break
        prompt = ex.get("question", ex.get("instruction", ""))
        if prompt and isinstance(prompt, str) and len(prompt.strip()) > 0:
            rows.append({"prompt": prompt.strip(), "malicious": False, "dataset_id": "openorca"})
    save(pd.DataFrame(rows), "openorca")


def download_dolly15k():
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    df = pd.DataFrame(ds)
    # Combine instruction + context if context exists
    prompts = []
    for _, row in df.iterrows():
        prompt = row["instruction"]
        if row.get("context") and isinstance(row["context"], str) and len(row["context"].strip()) > 0:
            prompt = f"{prompt}\n\nContext: {row['context']}"
        prompts.append(prompt)
    result = pd.DataFrame({
        "prompt": prompts,
        "malicious": False,
        "dataset_id": "dolly15k",
    })
    # Cap at 10k
    if len(result) > 10000:
        result = result.sample(n=10000, random_state=42).reset_index(drop=True)
    save(result, "dolly15k")


def download_10k_prompts():
    ds = load_dataset("fka/awesome-chatgpt-prompts", split="train")
    df = pd.DataFrame(ds)
    prompt_col = next(c for c in ["prompt", "text", "act"] if c in df.columns)
    result = pd.DataFrame({
        "prompt": df[prompt_col],
        "malicious": False,
        "dataset_id": "10k_prompts_ranked",
    })
    save(result, "10k_prompts_ranked")


def download_softage():
    # SoftAge-AI/prompt-eng_dataset — column is "Prompt" (capitalized)
    ds = load_dataset("SoftAge-AI/prompt-eng_dataset", split="train")
    df = pd.DataFrame(ds)
    result = pd.DataFrame({
        "prompt": df["Prompt"],
        "malicious": False,
        "dataset_id": "softage",
    })
    save(result, "softage")


def clone_bipia():
    target = LODO_DIR / "BIPIA"
    if target.exists():
        print(f"  BIPIA already cloned at {target}")
        return
    print("  Cloning BIPIA...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/microsoft/BIPIA", str(target)],
        check=True,
    )
    print(f"  BIPIA cloned to {target}")


def clone_injecagent():
    target = LODO_DIR / "InjecAgent"
    if target.exists():
        print(f"  InjecAgent already cloned at {target}")
        return
    print("  Cloning InjecAgent...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/uiuc-kang-lab/InjecAgent", str(target)],
        check=True,
    )
    print(f"  InjecAgent cloned to {target}")


def convert_bipia():
    """Convert BIPIA data to parquet after cloning.

    BIPIA has benign contexts (email/code/table JSONL) and attack strings (JSON).
    We save both: benign contexts as-is, and attack strings as malicious prompts.
    The cross-product (context + embedded attack) is what the original benchmark
    evaluates, but for probe training we keep them separate.
    """
    import json

    bipia_dir = LODO_DIR / "BIPIA" / "benchmark"
    if not bipia_dir.exists():
        print("  BIPIA not cloned, skipping conversion")
        return

    rows = []

    # Benign contexts from JSONL files
    for jsonl in bipia_dir.rglob("*.jsonl"):
        with open(jsonl) as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                # Context is the benign document; question is the legitimate query
                context = item.get("context", "")
                question = item.get("question", "")
                # Some contexts are lists/dicts (e.g. table data) — stringify them
                if not isinstance(context, str):
                    context = str(context)
                if not isinstance(question, str):
                    question = str(question)
                prompt = f"{context}\n\n{question}".strip() if question else context
                if prompt:
                    rows.append({"prompt": prompt, "malicious": False, "dataset_id": "bipia"})

    # Attack strings from JSON files
    for attack_file in bipia_dir.glob("*attack*.json"):
        with open(attack_file) as f:
            data = json.load(f)
        for category, items in data.items():
            for item in items:
                if isinstance(item, str) and item.strip():
                    rows.append({"prompt": item.strip(), "malicious": True, "dataset_id": "bipia"})

    if rows:
        save(pd.DataFrame(rows), "bipia")
    else:
        print("  BIPIA: no prompts extracted, may need manual inspection")


def convert_injecagent():
    """Convert InjecAgent data to parquet after cloning."""
    ia_dir = LODO_DIR / "InjecAgent"
    if not ia_dir.exists():
        print("  InjecAgent not cloned, skipping conversion")
        return
    # InjecAgent stores data as JSON files
    import json
    rows = []
    for json_path in ia_dir.rglob("*.json"):
        try:
            with open(json_path) as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    prompt = item.get("prompt", item.get("User Instruction", item.get("text", "")))
                    if prompt and isinstance(prompt, str):
                        # InjecAgent has both attack and benign examples
                        malicious = item.get("label", item.get("attack", True))
                        if isinstance(malicious, str):
                            malicious = malicious.lower() in ("true", "1", "yes", "malicious", "attack")
                        rows.append({"prompt": prompt, "malicious": bool(malicious), "dataset_id": "injecagent"})
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    if rows:
        save(pd.DataFrame(rows), "injecagent")
    else:
        print("  InjecAgent: no prompts extracted, may need manual inspection")


# Registry of all downloaders
HF_DATASETS = {
    "advbench": download_advbench,
    "harmbench": download_harmbench,
    "wildjailbreak": download_wildjailbreak,
    "yanismiraoui": download_yanismiraoui,
    "llmail": download_llmail,
    "mosscap": download_mosscap,
    "gandalf": download_gandalf,
    "jayavibhav": download_jayavibhav,
    "qualifire": download_qualifire,
    "safeguard": download_safeguard,
    "deepset": download_deepset,
    "enron": download_enron,
    "openorca": download_openorca,
    "dolly15k": download_dolly15k,
    "10k_prompts_ranked": download_10k_prompts,
    "softage": download_softage,
}

CLONE_DATASETS = {
    "bipia": (clone_bipia, convert_bipia),
    "injecagent": (clone_injecagent, convert_injecagent),
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download LODO benchmark datasets")
    parser.add_argument("--only", nargs="+", help="Download only these datasets")
    parser.add_argument("--skip-clones", action="store_true", help="Skip git clone datasets")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HuggingFace datasets")
    args = parser.parse_args()

    targets = args.only or list(HF_DATASETS.keys()) + list(CLONE_DATASETS.keys())

    if not args.skip_hf:
        for name, fn in HF_DATASETS.items():
            if name not in targets:
                continue
            print(f"\n[{name}]")
            try:
                fn()
            except Exception as e:
                print(f"  ERROR: {e}")

    if not args.skip_clones:
        for name, (clone_fn, convert_fn) in CLONE_DATASETS.items():
            if name not in targets:
                continue
            print(f"\n[{name}]")
            try:
                clone_fn()
                convert_fn()
            except Exception as e:
                print(f"  ERROR: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("LODO dataset summary:")
    parquets = sorted(LODO_DIR.glob("*.parquet"))
    total = 0
    for p in parquets:
        df = pd.read_parquet(p)
        total += len(df)
        n_mal = df["malicious"].sum()
        print(f"  {p.stem:25s}  {len(df):>6d} rows  ({n_mal:>5d} mal / {len(df)-n_mal:>5d} ben)")
    print(f"  {'TOTAL':25s}  {total:>6d} rows")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
