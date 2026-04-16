import pandas as pd
import hashlib
from pathlib import Path
from datetime import datetime


INPUT_PATH = Path("MASTER_HV_DATASET.csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

LOCKED_PATH = OUTPUT_DIR / "MASTER_HV_DATASET_STEP1_LOCKED.csv"
META_PATH = OUTPUT_DIR / "dataset_metadata.txt"


df = pd.read_csv(INPUT_PATH)
if df.empty:
    raise ValueError("Input dataset is empty")


required_columns = ["PROPERTY: HV", "Source"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

elem_cols = [c for c in df.columns if c.startswith("ELEM_")]
if not elem_cols:
    raise ValueError("No elemental composition columns found")


if df.isnull().values.any():
    raise ValueError("Dataset contains missing values")

if (df["PROPERTY: HV"] <= 0).any():
    raise ValueError("Non-positive hardness values detected")


composition_sum = df[elem_cols].sum(axis=1)
invalid_rows = df[abs(composition_sum - 1) > 1e-4]
if len(invalid_rows) > 0:
    raise ValueError(f"{len(invalid_rows)} rows failed composition normalization")


df_sorted = df.sort_values(by=["PROPERTY: HV", "Source"]).reset_index(drop=True)
df_sorted = df_sorted.sort_index(axis=1)


dataset_hash = hashlib.sha256(
    pd.util.hash_pandas_object(df_sorted, index=True).values
).hexdigest()


df_sorted.to_csv(LOCKED_PATH, index=False)


with open(META_PATH, "w") as f:
    f.write("Dataset metadata\n")
    f.write(f"Timestamp: {datetime.now()}\n")
    f.write(f"SHA256: {dataset_hash}\n")
    f.write(f"Samples: {len(df)}\n")
    f.write(f"Columns: {len(df.columns)}\n")
    f.write(f"Element features: {len(elem_cols)}\n\n")

    f.write("Columns:\n")
    for col in df.columns:
        f.write(f"{col}\n")
