"""Convert Geometry of Truth CSV files to Parquet format in data/got/."""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
GOT_DIR = DATA_DIR / "got"
GOT_DIR.mkdir(exist_ok=True)

csvs = sorted(DATA_DIR.glob("*.csv"))
print(f"Found {len(csvs)} CSV files in {DATA_DIR}")

for csv_path in csvs:
    df = pd.read_csv(csv_path)
    out = GOT_DIR / csv_path.with_suffix(".parquet").name
    df.to_parquet(out, index=False)
    print(f"  {csv_path.name} -> {out.name}  ({len(df)} rows, cols: {list(df.columns)})")

print(f"\nDone. {len(csvs)} parquet files written to {GOT_DIR}")
