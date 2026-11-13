import pandas as pd

df = pd.read_csv(
    "BMS_Datasets/combined_BMS_dataset.csv"
)

print("Dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nAttack distribution:")
print(df["Attack_Status"].value_counts())

print("\nOperating mode distribution:")
print(df["Operating_Mode"].value_counts())

print("\nScenario distribution:")
print(
    df.groupby(
        ["Operating_Mode", "Initial_SOC_pct", "Attack_Status"]
    ).agg(
        Samples=("Time_s", "size"),
        Runs=("Run_ID", "nunique")
    )
)

print("\nUnique simulation runs:")
print(df["Run_ID"].nunique())

print("\nDuplicate rows:", df.duplicated().sum())