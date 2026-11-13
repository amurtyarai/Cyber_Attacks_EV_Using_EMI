"""
EV BMS EMI Cyberattack Detection - Gaussian Naive Bayes
Model A: per-sample GaussianNB
Model B: windowed ripple-feature GaussianNB (50 samples = 0.1 s)

Evaluation: run-level 75/25 held-out test (train.csv / test.csv) and
leave-one-run-out (LORO) cross-validation over all 16 runs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.naive_bayes import GaussianNB

DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("BMS_Results_repro")
OUT_DIR.mkdir(exist_ok=True)

WINDOW = 50
A_FEATURES = ["Time_s", "Current_EMI_A", "Voltage_Battery_V",
              "SOC_BMS", "Mode_CHG", "Initial_SOC_pct"]
USE = ["Time_s", "Current_EMI_A", "Voltage_Battery_V", "SOC_BMS",
       "Attack_Status", "Operating_Mode", "Initial_SOC_pct", "Run_ID"]


def load(path):
    df = pd.read_csv(path, usecols=USE)
    df["Mode_CHG"] = (df["Operating_Mode"] == "CHG").astype(np.int8)
    return df


def scores(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def window_features(df, window=WINDOW):
    """Non-overlapping windows built separately inside each run."""
    X, y, rid = [], [], []
    for run, g in df.groupby("Run_ID", sort=False):
        cur = g["Current_EMI_A"].to_numpy()
        label = int(g["Attack_Status"].iloc[0])
        n = len(cur) // window
        w = cur[: n * window].reshape(n, window)
        d = np.diff(w, axis=1)
        X.append(np.column_stack([w.std(axis=1), np.ptp(w, axis=1), d.std(axis=1)]))
        y.append(np.full(n, label))
        rid.append(np.full(n, run, dtype=object))
    return np.vstack(X), np.concatenate(y), np.concatenate(rid)


def main():
    train = load(DATA_DIR / "train.csv")
    test = load(DATA_DIR / "test.csv")
    results = {}

    # ---------------- Model A: per-sample ----------------
    gnb = GaussianNB().fit(train[A_FEATURES], train["Attack_Status"])
    pred = gnb.predict(test[A_FEATURES])
    results["A_per_sample"] = {"features": A_FEATURES,
                               "test": scores(test["Attack_Status"], pred)}
    per_run = (test.assign(pred=pred).groupby("Run_ID")
               .agg(True_label=("Attack_Status", "first"),
                    Samples=("pred", "size"),
                    Pct_predicted_attack=("pred", lambda s: 100 * s.mean())))
    per_run["Accuracy_pct"] = np.where(per_run.True_label == 1,
                                       per_run.Pct_predicted_attack,
                                       100 - per_run.Pct_predicted_attack)
    per_run.to_csv(OUT_DIR / "per_run_test_model_A.csv")

    # ---------------- Model B: windowed ----------------
    Xtr, ytr, _ = window_features(train)
    Xte, yte, rte = window_features(test)
    gnb_b = GaussianNB().fit(Xtr, ytr)
    pred_b = gnb_b.predict(Xte)
    results["B_windowed"] = {"features": ["cur_std", "cur_ptp", "diff_std"],
                             "window_samples": WINDOW,
                             "test": scores(yte, pred_b)}
    pd.DataFrame({"Run_ID": rte, "y": yte, "pred": pred_b}).groupby("Run_ID").agg(
        True_label=("y", "first"), Windows=("y", "size"),
        Pct_predicted_attack=("pred", lambda s: 100 * s.mean())
    ).to_csv(OUT_DIR / "per_run_test_model_B.csv")
    means = pd.DataFrame(Xte if False else np.vstack([Xtr, Xte]),
                         columns=["cur_std", "cur_ptp", "diff_std"])
    means["y"] = np.concatenate([ytr, yte])
    means.groupby("y").mean().rename(index={0: "Normal", 1: "Attack"}).round(3) \
        .to_csv(OUT_DIR / "model_B_class_means.csv")

    # ---------------- LORO over all 16 runs ----------------
    full = pd.concat([train, test], ignore_index=True)
    runs = full["Run_ID"].unique()
    # Model A
    yt, yp = [], []
    for r in runs:
        tr, te = full[full.Run_ID != r], full[full.Run_ID == r]
        m = GaussianNB().fit(tr[A_FEATURES], tr["Attack_Status"])
        yt.append(te["Attack_Status"].to_numpy())
        yp.append(m.predict(te[A_FEATURES]))
    results["A_per_sample"]["loro_cv"] = scores(np.concatenate(yt), np.concatenate(yp))
    # Model B
    Xall, yall, rall = window_features(full)
    yt, yp = [], []
    for r in runs:
        tr, te = rall != r, rall == r
        m = GaussianNB().fit(Xall[tr], yall[tr])
        yt.append(yall[te]); yp.append(m.predict(Xall[te]))
    results["B_windowed"]["loro_cv"] = scores(np.concatenate(yt), np.concatenate(yp))

    (OUT_DIR / "metrics_repro.json").write_text(json.dumps(results, indent=2))
    for k, v in results.items():
        for split in ("test", "loro_cv"):
            s = v[split]
            print(f"{k:14s} {split:8s} acc={s['accuracy']:.4f} prec={s['precision']:.4f} "
                  f"rec={s['recall']:.4f} f1={s['f1']:.4f} cm={s['confusion_matrix']}")


if __name__ == "__main__":
    main()
