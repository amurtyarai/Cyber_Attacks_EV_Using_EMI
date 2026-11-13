
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "BMS_Datasets" / "combined_BMS_dataset.csv"

OUTPUT_DIR = BASE_DIR / "BMS_Splits"

TEST_SIZE = 0.25
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================
# 2. LOAD DATASET
# ==========================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)

required_columns = [
    "Time_s",
    "Attack_Status",
    "Operating_Mode",
    "Initial_SOC_pct",
    "Run_ID"
]

if not set(required_columns).issubset(df.columns):
    raise ValueError("Required columns are missing.")

if df.isnull().values.any():
    raise ValueError("Missing values found in dataset.")

df["Run_ID"] = df["Run_ID"].astype(str)

# ==========================================
# 3. CREATE ONE RECORD PER SIMULATION RUN
# ==========================================

# Every Run_ID must belong to one scenario only.

run_checks = df.groupby("Run_ID").agg(
    mode_count=("Operating_Mode", "nunique"),
    attack_count=("Attack_Status", "nunique"),
    soc_count=("Initial_SOC_pct", "nunique")
)

if (run_checks > 1).any().any():
    raise ValueError(
        "A Run_ID contains inconsistent scenario metadata."
    )

run_meta = (
    df.groupby("Run_ID", as_index=False)
    .agg(
        Operating_Mode=("Operating_Mode", "first"),
        Attack_Status=("Attack_Status", "first"),
        Initial_SOC_pct=("Initial_SOC_pct", "first"),
        Samples=("Time_s", "size")
    )
)

print("\nTotal simulation runs:", len(run_meta))

# ==========================================
# 4. CREATE STRATIFICATION GROUPS
# ==========================================

# Preserve balance across:
# CHG + Normal
# CHG + Attack
# DIS + Normal
# DIS + Attack

run_meta["Stratum"] = (
    run_meta["Operating_Mode"].astype(str)
    + "_Attack_"
    + run_meta["Attack_Status"].astype(str)
)

print("\nRuns per stratum:")
print(run_meta["Stratum"].value_counts())

if run_meta["Stratum"].value_counts().min() < 2:
    raise ValueError(
        "Not enough simulation runs in a stratum to split."
    )

# ==========================================
# 5. SPLIT AT RUN LEVEL
# ==========================================

train_runs, test_runs = train_test_split(
    run_meta,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=run_meta["Stratum"]
)

train_ids = set(train_runs["Run_ID"])
test_ids = set(test_runs["Run_ID"])

# Check that no Run_ID appears in both sets
if train_ids.intersection(test_ids):
    raise ValueError("Data leakage: Run_ID overlap detected.")

# ==========================================
# 6. SELECT ALL ROWS FOR EACH RUN
# ==========================================

train_df = df[df["Run_ID"].isin(train_ids)].copy()

test_df = df[df["Run_ID"].isin(test_ids)].copy()

# Sort samples by run and time
train_df = train_df.sort_values(
    ["Run_ID", "Time_s"]
).reset_index(drop=True)

test_df = test_df.sort_values(
    ["Run_ID", "Time_s"]
).reset_index(drop=True)

# ==========================================
# 7. VALIDATE THE SPLIT
# ==========================================

assert set(train_df["Run_ID"]).isdisjoint(
    set(test_df["Run_ID"])
)

assert (
    len(train_df) + len(test_df) == len(df)
)

print("\n========== SPLIT SUMMARY ==========")

print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)

print("\nTraining attack distribution:")
print(train_df["Attack_Status"].value_counts())

print("\nTesting attack distribution:")
print(test_df["Attack_Status"].value_counts())

print("\nTraining scenario distribution:")
print(
    train_df.groupby(
        ["Operating_Mode", "Attack_Status"]
    ).agg(
        Samples=("Time_s", "size"),
        Runs=("Run_ID", "nunique")
    )
)

print("\nTesting scenario distribution:")
print(
    test_df.groupby(
        ["Operating_Mode", "Attack_Status"]
    ).agg(
        Samples=("Time_s", "size"),
        Runs=("Run_ID", "nunique")
    )
)

# ==========================================
# 8. CREATE SPLIT MANIFEST
# ==========================================

train_manifest = run_meta[
    run_meta["Run_ID"].isin(train_ids)
].copy()

train_manifest["Partition"] = "TRAIN"

test_manifest = run_meta[
    run_meta["Run_ID"].isin(test_ids)
].copy()

test_manifest["Partition"] = "TEST"

manifest = pd.concat(
    [train_manifest, test_manifest],
    ignore_index=True
)

manifest = manifest.sort_values(
    ["Partition", "Operating_Mode", "Attack_Status", "Run_ID"]
)

# ==========================================
# 9. SAVE OUTPUT FILES
# ==========================================

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)

manifest.to_csv(
    OUTPUT_DIR / "split_manifest.csv",
    index=False
)

print("\nFiles successfully saved:")
print(OUTPUT_DIR / "train.csv")
print(OUTPUT_DIR / "test.csv")
print(OUTPUT_DIR / "split_manifest.csv")

print("\nTrain runs:", len(train_ids))
print("Test runs:", len(test_ids))

print("\nNo Run_ID leakage detected.")
print("Dataset splitting completed successfully.")
