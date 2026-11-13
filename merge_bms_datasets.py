
from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_FOLDER = Path("BMS_Datasets")

OUTPUT_FILE = DATASET_FOLDER / "combined_BMS_dataset.csv"

EXPECTED_COLUMNS = [
    "Time_s",
    "Current_EMI_A",
    "Current_Battery_A",
    "Voltage_Battery_V",
    "Voltage_Induced_V",
    "SOC_BMS",
    "SOC_True",
    "Resistance_Ohm",
    "Attack_Status",
    "Operating_Mode",
    "Initial_SOC_pct",
    "Capacity_Ah",
    "BMS_SampleTime_s",
    "Dataset_SampleTime_s",
    "EMI_Frequency_Hz",
    "EMI_SourcePeak_V",
    "EMI_Phase_deg",
    "Run_ID"
]

# ==========================================
# 2. FIND CSV FILES
# ==========================================

csv_files = sorted(
    f for f in DATASET_FOLDER.glob("*.csv")
    if f.name != OUTPUT_FILE.name
    and not f.stem.lower().startswith("combined")
)

if not csv_files:
    raise FileNotFoundError("No scenario CSV files found.")

print(f"Found {len(csv_files)} CSV files.")

# ==========================================
# 3. READ AND VALIDATE EACH FILE
# ==========================================

all_data = []
existing_run_ids = set()

for file in csv_files:

    print(f"\nProcessing: {file.name}")

    df = pd.read_csv(file)

    # Check columns
    if set(df.columns) != set(EXPECTED_COLUMNS):
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        extra = set(df.columns) - set(EXPECTED_COLUMNS)

        raise ValueError(
            f"Column mismatch in {file.name}\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )

    # Maintain consistent column order
    df = df[EXPECTED_COLUMNS]

    # Check missing values
    if df.isnull().values.any():
        raise ValueError(f"Missing values found in {file.name}")

    # Each CSV should represent exactly one simulation
    run_ids = df["Run_ID"].astype(str).unique()

    if len(run_ids) != 1:
        raise ValueError(
            f"Expected one Run_ID in {file.name}, "
            f"found {len(run_ids)}"
        )

    run_id = run_ids[0]

    # Prevent duplicate simulation runs
    if run_id in existing_run_ids:
        raise ValueError(f"Duplicate Run_ID detected: {run_id}")

    existing_run_ids.add(run_id)

    # Check attack labels
    if not set(df["Attack_Status"].unique()).issubset({0, 1}):
        raise ValueError(f"Invalid Attack_Status in {file.name}")

    # Check operating modes
    if not set(df["Operating_Mode"].unique()).issubset(
        {"CHG", "DIS"}
    ):
        raise ValueError(f"Invalid operating mode in {file.name}")

    # Check sampling interval
    time_diff = np.diff(df["Time_s"].to_numpy())

    if not np.allclose(time_diff, 0.002, atol=1e-8):
        raise ValueError(
            f"Incorrect sampling interval in {file.name}"
        )

    # Check that time is ordered
    if not np.all(time_diff > 0):
        raise ValueError(f"Time is not increasing in {file.name}")

    all_data.append(df)

    print(f"  Run ID: {run_id}")
    print(f"  Rows: {len(df)}")
    print(f"  Attack label: {df['Attack_Status'].iloc[0]}")
    print(f"  Mode: {df['Operating_Mode'].iloc[0]}")

# ==========================================
# 4. MERGE ALL DATASETS
# ==========================================

combined_df = pd.concat(
    all_data,
    ignore_index=True
)

# ==========================================
# 5. FINAL VALIDATION
# ==========================================

print("\n========== MERGED DATASET ==========")

print("Shape:", combined_df.shape)

print(
    "Missing values:",
    combined_df.isnull().sum().sum()
)

print(
    "Number of simulation runs:",
    combined_df["Run_ID"].nunique()
)

print("\nAttack distribution:")
print(combined_df["Attack_Status"].value_counts())

print("\nOperating mode distribution:")
print(combined_df["Operating_Mode"].value_counts())

print("\nScenario summary:")
print(
    combined_df.groupby(
        ["Operating_Mode", "Attack_Status"]
    ).agg(
        Rows=("Time_s", "size"),
        Runs=("Run_ID", "nunique")
    )
)

# ==========================================
# 6. SAVE MERGED DATASET
# ==========================================

combined_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nMerged dataset saved successfully!")
print("File:", OUTPUT_FILE)
print("Final shape:", combined_df.shape)
