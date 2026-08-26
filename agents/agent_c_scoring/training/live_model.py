"""
TWO-MODEL SEPARATION RULE (do not violate):
This is the LIVE model. It trains ONLY on Person A's synthetic archetype
labels (officer-level D1-D6 features -> clean=0 / colluding-fraudulent=1).
It NEVER touches Kaggle fraud_reported data or claim-level features.
See training/validation_only_model.py for the separate validation-only model.
These two models share zero training data, zero code paths.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import json
import joblib
from pathlib import Path

RAW_CSV = Path("data_utils/raw_data/officer_twins_live.csv")
RESULTS_DIR = Path("agents/agent_c_scoring/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "d1_approval_rate", "d1_approval_rate_z",
    "d2_garage_concentration", "d2_garage_concentration_z",
    "d3_threshold_proximity", "d3_threshold_proximity_z",
    "d4_surveyor_diversity", "d4_surveyor_diversity_z",
    "d5_turnaround_variance", "d5_turnaround_variance_z",
    "d6_after_hours_pct", "d6_after_hours_pct_z",
]

def build_label(df: pd.DataFrame) -> pd.Series:
    label = pd.Series(0, index=df.index, dtype=int)

    label[df["archetype"] == "actively_fraudulent"] = 1

    gradual_mask = df["archetype"] == "gradually_colluding"
    past_start = gradual_mask & df["collusion_start_day"].notna() & (
        df["day"] >= df["collusion_start_day"]
    )
    label[past_start] = 1

    return label

def main():
    print(f"Loading {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV)
    print(f"Loaded shape: {df.shape}")

    df["label"] = build_label(df)

    print("\n=== Label distribution ===")
    print(df["label"].value_counts())
    print("\n=== Per-archetype label check ===")
    print(df.groupby("archetype")["label"].value_counts())

    X = df[FEATURE_COLS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining XGBoost live model...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n=== Live model evaluation (held-out test set) ===")
    print(classification_report(y_test, y_pred))
    print(f"AUC: {auc:.4f}")
    print(f"Confusion matrix: {cm}")

    metrics = {
        "classification_report": report,
        "auc": auc,
        "confusion_matrix": cm,
        "feature_cols": FEATURE_COLS,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(RESULTS_DIR / "live_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(model, RESULTS_DIR / "live_model.pkl")

    print(f"\nSaved: {RESULTS_DIR / 'live_model_metrics.json'}")
    print(f"Saved: {RESULTS_DIR / 'live_model.pkl'}")

if __name__ == "__main__":
    main()
