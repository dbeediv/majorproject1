# VALIDATION-ONLY MODEL: trains on Kaggle fraud_reported label, claim-level features.
# Exists ONLY to produce paper metrics (precision/recall/F1/AUC). NEVER runs live.
# This model is never called by the live pipeline (Agent 3 / live_model.py is separate
# and trains on Person A's synthetic archetype labels instead — see Fix C-1).
#
# v2: uses 5-fold stratified cross-validation + bootstrap confidence intervals
# instead of a single train/test split, since the Kaggle dataset is small (n=1000).

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

# Reuse Person A's loader directly — do not re-implement cleaning logic here.
sys.path.append(str(Path(__file__).resolve().parents[3] / "data_utils"))
from loader import load_and_clean_kaggle_data  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = [
    "months_as_customer", "age", "policy_deductable", "policy_annual_premium",
    "umbrella_limit", "capital-gains", "capital-loss", "incident_hour_of_the_day",
    "number_of_vehicles_involved", "bodily_injuries", "witnesses",
    "total_claim_amount", "auto_year",
]

CATEGORICAL_FEATURES = [
    "policy_csl", "insured_sex", "insured_education_level", "insured_occupation",
    "insured_hobbies", "insured_relationship", "incident_type", "collision_type",
    "incident_severity", "authorities_contacted", "property_damage",
    "police_report_available",
]

LABEL_COL = "fraud_reported"
N_FOLDS = 5
N_BOOTSTRAP = 1000
RANDOM_STATE = 42


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("Unknown").astype(str)
        X[col] = LabelEncoder().fit_transform(X[col])
    y = df[LABEL_COL]
    return X, y


def make_model(scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
    )


def bootstrap_ci(y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray,
                  metric_fn, n_boot: int = N_BOOTSTRAP, seed: int = RANDOM_STATE) -> tuple[float, float]:
    """95% CI via resampling the test set with replacement."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue  # skip degenerate resamples with only one class
        try:
            scores.append(metric_fn(yt, yp, ypr))
        except ValueError:
            continue
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def cross_validate(X: pd.DataFrame, y: pd.Series) -> dict:
    """5-fold stratified CV — reports mean +/- std across folds instead of
    a single split, since n=1000 is small enough that one split can be
    lucky or unlucky."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc_roc": []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        model = make_model(scale_pos_weight=n_neg / n_pos)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        fold_metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        fold_metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        fold_metrics["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        fold_metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        fold_metrics["auc_roc"].append(roc_auc_score(y_test, y_proba))

        print(f"  Fold {fold}/{N_FOLDS}: AUC-ROC={fold_metrics['auc_roc'][-1]:.4f}  "
              f"F1={fold_metrics['f1'][-1]:.4f}")

    cv_summary = {}
    for k, v in fold_metrics.items():
        cv_summary[k] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "folds": v}
    return cv_summary


def holdout_with_bootstrap_ci(X: pd.DataFrame, y: pd.Series) -> dict:
    """A single held-out split, but with bootstrap CIs instead of point
    estimates — this is the number to actually quote in the paper
    ('AUC-ROC = 0.82, 95% CI [x, y]')."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    model = make_model(scale_pos_weight=n_neg / n_pos)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_test_arr = y_test.to_numpy()

    point = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_proba),
    }

    ci = {
        "auc_roc": bootstrap_ci(y_test_arr, y_proba, y_pred, lambda yt, yp, ypr: roc_auc_score(yt, ypr)),
        "f1": bootstrap_ci(y_test_arr, y_proba, y_pred, lambda yt, yp, ypr: f1_score(yt, yp, zero_division=0)),
        "precision": bootstrap_ci(y_test_arr, y_proba, y_pred, lambda yt, yp, ypr: precision_score(yt, yp, zero_division=0)),
        "recall": bootstrap_ci(y_test_arr, y_proba, y_pred, lambda yt, yp, ypr: recall_score(yt, yp, zero_division=0)),
    }

    cm = confusion_matrix(y_test, y_pred)
    confusion = {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}

    joblib.dump(model, RESULTS_DIR / "validation_model.pkl")

    return {"point_estimate": point, "confidence_intervals_95": ci, "confusion_matrix": confusion}


def train_and_evaluate(data_path: str | None = None) -> dict:
    df = load_and_clean_kaggle_data(data_path) if data_path else load_and_clean_kaggle_data()
    X, y = build_feature_frame(df)

    print(f"=== 5-Fold Stratified Cross-Validation (n={len(X)}) ===")
    cv_results = cross_validate(X, y)

    print(f"\n=== Held-out split with bootstrap 95% CI (n_boot={N_BOOTSTRAP}) ===")
    holdout_results = holdout_with_bootstrap_ci(X, y)

    results = {
        "dataset_size": len(X),
        "fraud_rate": float(y.mean()),
        "cross_validation": cv_results,
        "holdout_with_ci": holdout_results,
    }

    with open(RESULTS_DIR / "validation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    results = train_and_evaluate()

    print("\n" + "=" * 55)
    print("SUMMARY — for paper methodology section")
    print("=" * 55)
    print(f"Dataset: n={results['dataset_size']}, fraud rate={results['fraud_rate']:.3f}")
    print()
    print("5-fold CV (mean ± std across folds):")
    for metric, vals in results["cross_validation"].items():
        print(f"  {metric:10s}: {vals['mean']:.4f} ± {vals['std']:.4f}")
    print()
    print("Held-out split, point estimate + 95% bootstrap CI:")
    pe = results["holdout_with_ci"]["point_estimate"]
    ci = results["holdout_with_ci"]["confidence_intervals_95"]
    for metric in ["auc_roc", "precision", "recall", "f1"]:
        lo, hi = ci[metric]
        print(f"  {metric:10s}: {pe[metric]:.4f}  [95% CI: {lo:.4f}, {hi:.4f}]")
    print()
    print(f"Confusion matrix (held-out): {results['holdout_with_ci']['confusion_matrix']}")
    print()
    print("Saved: results/validation_model.pkl, results/validation_metrics.json")