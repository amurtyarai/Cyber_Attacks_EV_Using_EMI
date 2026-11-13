# EV BMS EMI Cyberattack Detection Dataset

## 1. Overview

This repository contains a simulated dataset developed for a final-year project on **electromagnetic interference (EMI) attacks against an Electric Vehicle (EV) Battery Management System (BMS)**.

The experiments use MATLAB/Simulink (R2024b) to simulate normal and attacked battery operation in two operating modes:

- **CHG** — charging
- **DIS** — discharging

In the attack scenarios, a 50 Hz sinusoidal source is coupled into the current-sensing circuit. The BMS estimates State of Charge (SOC) using current-based Coulomb counting. The dataset records the physical battery measurements, the current measurement supplied to the BMS, SOC estimates, and scenario metadata.

The main file is:

```text
combined_BMS_dataset.csv
```

## 2. Dataset summary

| Property | Value |
|---|---:|
| Total rows | **800,016** |
| Columns | **18** |
| Simulation runs | **16** |
| Normal samples (`Attack_Status = 0`) | **400,008** |
| Attack samples (`Attack_Status = 1`) | **400,008** |
| Normal / attack ratio | **1:1** |
| Charging samples | **400,008** |
| Discharging samples | **400,008** |
| Samples per 100-second simulation | **50,001** |
| Export sampling frequency | **500 Hz** |
| Export sample interval | **0.002 s** |
| BMS SOC/current-estimator sample interval | **0.1 s** |
| Missing values in the audited merged CSV | **0** |
| Duplicate rows in the audited merged CSV | **0** |

Each simulation covers 100 seconds. The first sample is at 0 seconds and the final sample is at 100 seconds, giving 50,001 samples at 500 Hz.

### Class and operating-mode distribution

| Operating mode | Label | Meaning | Runs | Samples |
|---|---:|---|---:|---:|
| CHG | 0 | Normal charging | 4 | 200,004 |
| CHG | 1 | Charging under EMI attack | 4 | 200,004 |
| DIS | 0 | Normal discharging | 4 | 200,004 |
| DIS | 1 | Discharging under EMI attack | 4 | 200,004 |
| **Total** | | | **16** | **800,016** |

The dataset is balanced overall and within each operating mode.

## 3. Scenario coverage

The current dataset covers the following operating-mode and nominal initial-SOC combinations:

| Operating mode | Nominal initial SOC | Normal runs | Attack runs |
|---|---:|---:|---:|
| CHG | 10% | 1 | 2 |
| CHG | 30% | 1 | 2 |
| CHG | 50% | 1 | 0 |
| CHG | 70% | 1 | 0 |
| DIS | 30% | 1 | 0 |
| DIS | 50% | 1 | 0 |
| DIS | 70% | 1 | 2 |
| DIS | 90% | 1 | 2 |

**Coverage limitation:** Attack runs are not available for every initial-SOC/operating-mode combination. In particular, the current dataset does not include attacks for CHG at 50% or 70% initial SOC, or DIS at 30% or 50% initial SOC. Keep this limitation in mind when interpreting model performance and generalisation.

## 4. Column descriptions

The merged CSV contains the following 18 columns. The original names and column order are retained in the file.

| Column | Unit / type | Description |
|---|---|---|
| `Time_s` | seconds | Time within an individual simulation run. It resets at the start of each run. |
| `Current_EMI_A` | A | Current measurement supplied to the BMS/current-estimation path. In attack runs, this is the current after EMI injection; in normal runs, it equals `Current_Battery_A`. |
| `Current_Battery_A` | A | Physical/reference battery current logged from the battery circuit. |
| `Voltage_Battery_V` | V | Battery terminal voltage. |
| `Voltage_Induced_V` | V | Voltage measured in the EMI coupling/injection circuit. It is zero in normal runs because the EMI circuit is absent/disconnected. |
| `SOC_BMS` | % | SOC calculated by the BMS using its current-based estimation logic. |
| `SOC_True` | % | Reference SOC from the simulation, retained for evaluation and comparison. |
| `Resistance_Ohm` | ohm | External load resistance metadata; configured as 3 ohms in these scenarios. |
| `Attack_Status` | binary | Target label: `0` = normal operation; `1` = EMI attack. |
| `Operating_Mode` | category | `CHG` = charging; `DIS` = discharging. |
| `Initial_SOC_pct` | % | Nominal initial SOC configured for the scenario. The first logged SOC sample may differ slightly due to model initialisation. |
| `Capacity_Ah` | Ah | Battery capacity metadata; 292.4 Ah. |
| `BMS_SampleTime_s` | seconds | BMS SOC/current-estimator sample interval; 0.1 s. |
| `Dataset_SampleTime_s` | seconds | Exported dataset sample interval; 0.002 s. |
| `EMI_Frequency_Hz` | Hz | EMI source frequency metadata. It is 50 Hz for attack runs and 0 for normal runs. |
| `EMI_SourcePeak_V` | V | Configured AC source peak voltage metadata. It is 230 V for attack runs and 0 for normal runs. This is **not** the measured induced voltage. |
| `EMI_Phase_deg` | degrees | Configured EMI source phase metadata. Normal runs use 0 as a placeholder because no EMI source is present. |
| `Run_ID` | string | Unique identifier for the simulation run. Use this to group rows and prevent train/test leakage. |

## 5. Simulation and signal-processing notes

### Normal scenarios

The normal-operation model has no EMI-inducing circuit. The export process therefore records:

- `Attack_Status = 0`
- `Current_EMI_A = Current_Battery_A`
- `Voltage_Induced_V = 0`
- `EMI_Frequency_Hz = 0`
- `EMI_SourcePeak_V = 0`
- `EMI_Phase_deg = 0` (placeholder metadata)

### Attack scenarios

The attack scenarios use a 50 Hz sinusoidal source with a configured source peak voltage of 230 V. The EMI circuit is connected for the attack simulations. The actual voltage measured in the coupling circuit is recorded in `Voltage_Induced_V`; it should not be assumed to equal the source's configured 230 V peak.

The phase angle is included in the metadata because it can affect the disturbance observed by the BMS. The BMS estimator uses a 0.1-second sample interval (10 Hz), while the exported electrical signals are sampled at 500 Hz. As 50 Hz is an integer multiple of 10 Hz, sampling alignment and phase can strongly affect the disturbance values seen by the estimator. This is a characteristic of the simulated setup and should be considered when interpreting results.

### Export sampling

The electrical signals were logged by the Simulink model at solver-dependent timestamps. The export workflow interpolated electrical signals to a 2 kHz uniform grid, applied anti-alias filtering and downsampled to 500 Hz. SOC signals, logged at the BMS sample interval, were aligned to the exported time grid using previous-value interpolation. Consequently, each SOC sample is typically repeated across multiple 500 Hz rows until the next BMS update.

The 500 Hz export provides 10 samples per 50 Hz EMI cycle. The sampling and filtering process is part of the dataset-generation pipeline and should be kept consistent when creating additional scenarios.

## 6. Recommended use for machine-learning experiments

### Target

Use `Attack_Status` as the classification target:

- `0` — normal
- `1` — EMI attack

### Candidate input features

The appropriate input features depend on the sensor signals available to the intended deployed detector. Candidate measurements include:

- `Current_EMI_A`
- `Voltage_Battery_V`
- `SOC_BMS`
- `Voltage_Induced_V`, **only if this measurement would actually be available to the proposed detector**

Use engineering judgement to define the detector's available signals. In particular:

- Do not use `Attack_Status` as an input feature; it is the label.
- Do not use `SOC_True` as an input feature. It is a simulation reference used for evaluation, not a normal BMS measurement available in deployment.
- `Current_Battery_A` is a reference/physical current measurement. Exclude it from model inputs if the intended detector cannot access an independent trusted battery-current measurement.
- Do not use `Run_ID` as a model input. It is an identifier for grouping and data splitting.
- Scenario configuration columns such as EMI frequency, source amplitude and phase are metadata. Avoid using them as model inputs if they would reveal the attack configuration or would not be available to a deployed detector.

### Train/test splitting and leakage prevention

Rows from the same simulation run are strongly related in time. **Do not randomly split individual rows** into training and test sets. Instead, split by `Run_ID`, assigning every row from one run to only one partition.

For the current 16-run dataset, a stratified 75/25 group split can place three runs from each of the four groups (CHG/normal, CHG/attack, DIS/normal, DIS/attack) in training and one run from each group in testing. This gives:

| Partition | Runs | Samples |
|---|---:|---:|
| Training | 12 | 600,012 |
| Testing | 4 | 200,004 |

This is a recommended split strategy, not a pre-generated split. If a separate validation set is needed for model selection, create it from training runs only, again grouping by `Run_ID`. Do not use the test set for tuning.

Fit scaling/normalisation parameters using training data only, then apply the same fitted transformation to validation and test data. For CNN/LSTM or other sequence models, split by run first and create time-series windows independently within each partition; never create windows that cross run boundaries.

## 7. Loading the dataset

### Python / Pandas

```python
import pandas as pd

file_path = "BMS_Datasets/combined_BMS_dataset.csv"
df = pd.read_csv(file_path)

print(df.shape)
print(df.head())
print(df["Attack_Status"].value_counts())
print(df["Operating_Mode"].value_counts())
print("Unique runs:", df["Run_ID"].nunique())
```

Expected shape:

```text
(800016, 18)
```

### MATLAB

```matlab
T = readtable("BMS_Datasets/combined_BMS_dataset.csv");

size(T)
unique(T.Attack_Status)
unique(T.Operating_Mode)
numel(unique(T.Run_ID))
```

## 8. File and reproducibility notes

Recommended shared-project structure:

```text
EV_Cyber_Attack_Shared/
|
|-- README.md
|-- Dataset/
|   |-- combined_BMS_dataset.csv
|   |-- train.csv                 % optional generated split
|   |-- test.csv                  % optional generated split
|   `-- split_manifest.csv        % optional run-to-partition mapping
|
`-- Scripts/
    |-- merge_bms_datasets.py
    |-- split_bms_dataset.py
    `-- audit.py
```

The merged dataset was created by combining 16 scenario CSV files. The audit performed on the merged file reported:

- Shape: 800,016 rows × 18 columns
- Missing values: 0
- Duplicate rows: 0
- Unique simulation runs: 16
- Attack labels: 400,008 normal and 400,008 attack samples
- Charging/discharging: 400,008 samples each

For reproducibility, share the merge and split scripts with the CSV. Preserve each run's `Run_ID` and the original scenario files where storage allows. If the dataset is regenerated or modified, update this README and rerun the audit.

## 9. Limitations and interpretation

1. **Simulation data:** These are MATLAB/Simulink simulation results, not measurements collected from a physical EV or production BMS.
2. **Limited independent runs:** The dataset contains 16 simulation runs. Although it has 800,016 rows, the number of independent simulation scenarios is much smaller than the row count suggests.
3. **Incomplete SOC/mode attack coverage:** Some initial-SOC and operating-mode combinations have normal data but no corresponding attack data; see the scenario coverage table.
4. **Specific attack configuration:** The attack data represent the simulated 50 Hz EMI injection setup and configured circuit parameters. Results should not be generalised to all EMI or automotive cyberattacks without further testing.
5. **Sampling alignment:** The 50 Hz disturbance and 10 Hz BMS estimator sampling have a precise frequency relationship. Different timing, phase, sampling rates or hardware behaviour may produce different observed effects.
6. **Balanced classes do not imply balanced operating conditions:** Overall class balance is 1:1, but the attack coverage across initial SOC values is not uniform.
7. **No claim of real-world accuracy:** Model performance on this dataset alone does not establish equivalent performance on physical EV systems. Independent simulations, varied attack parameters and hardware/vehicle validation would be needed for stronger claims.

## 10. Dataset purpose

This dataset is intended for academic exploration of:

- Detection of EMI-induced current-sensor disturbances.
- Classification of normal and attacked BMS operation.
- Comparison of detection behaviour during charging and discharging.
- Analysis of SOC estimation error under current-sensor disturbance.
- Development and evaluation of machine-learning and deep-learning models using time-series sensor data.

Please cite or acknowledge the project team and describe the dataset as **simulation-generated EV BMS EMI attack data** when using it in reports, presentations or publications.
