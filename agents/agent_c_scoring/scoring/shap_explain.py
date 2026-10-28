"""
SHAP explainability for the LIVE model only.
Uses TreeExplainer on the uncalibrated base XGBoost model from
training/live_model.py (results/live_model.pkl).
Outputs top_3_shap in the exact shape required by contracts.ShapContribution.
"""

import json
import joblib
import shap
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("agents/agent_c_scoring/results/live_model.pkl")

FEATURE_COLS = [
    "d1_approval_rate", "d1_approval_rate_z",
    "d2_garage_concentration", "d2_garage_concentration_z",
    "d3_threshold_proximity", "d3_threshold_proximity_z",
    "d4_surveyor_diversity", "d4_surveyor_diversity_z",
    "d5_turnaround_variance", "d5_turnaround_variance_z",
    "d6_after_hours_pct", "d6_after_hours_pct_z",
]

DIM_LABELS = {
    "d1": "Approval Rate",
    "d2": "Garage Concentration",
    "d3": "Threshold Proximity",
    "d4": "Surveyor Diversity",
    "d5": "Turnaround Variance",
    "d6": "After-Hours Activity",
}

_model = None
_explainer = None


def _load():
    global _model, _explainer
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        booster = _model.get_booster()

        # --- patch base_score serialization bug (XGBoost 3.x + shap) ---
        # The bug ONLY affects JSON-format serialization (base_score gets
        # wrapped as a string-encoded list, e.g. "[2.13e-1]"). UBJSON
        # (binary) format is unaffected. So we only patch the JSON path
        # and fall through to the real save_raw for everything else.
        _orig_save_raw = booster.save_raw

        raw = _orig_save_raw(raw_format="json")
        model_json = json.loads(raw)
        bad = model_json["learner"]["learner_model_param"]["base_score"]
        fixed = str(float(str(bad).strip("[]")))
        model_json["learner"]["learner_model_param"]["base_score"] = fixed
        fixed_json_bytes = json.dumps(model_json).encode("utf-8")

        def _patched_save_raw(raw_format="deprecated"):
            if raw_format in ("json", "deprecated_json"):
                return fixed_json_bytes
            return _orig_save_raw(raw_format=raw_format)

        booster.save_raw = _patched_save_raw
        # ----------------------------------------------------------------

        _explainer = shap.TreeExplainer(_model)
    return _model, _explainer


def get_top_3_shap(feature_row: pd.Series) -> list[dict]:
    """
    feature_row: a pandas Series indexed by FEATURE_COLS (one officer, one day).
    Returns a list of exactly 3 dicts matching contracts.ShapContribution:
        dim_code, dim_label, raw_value, shap_contribution_pts
    Ranked by absolute SHAP contribution, largest first.
    """
    model, explainer = _load()

    X = feature_row[FEATURE_COLS].to_frame().T.apply(pd.to_numeric, errors="coerce")
    shap_values = explainer.shap_values(X)[0]

    contributions = []
    for col, sv in zip(FEATURE_COLS, shap_values):
        dim_code = col.split("_")[0]
        raw_col = dim_code + "_" + "_".join(col.split("_")[1:]).replace("_z", "")
        contributions.append({
            "dim_code": dim_code,
            "dim_label": DIM_LABELS[dim_code],
            "raw_value": float(feature_row[raw_col]) if raw_col in feature_row else float(feature_row[col]),
            "shap_contribution_pts": round(float(sv) * 100, 2),
            "_abs": abs(sv),
        })

    best_per_dim = {}
    for c in contributions:
        d = c["dim_code"]
        if d not in best_per_dim or c["_abs"] > best_per_dim[d]["_abs"]:
            best_per_dim[d] = c

    ranked = sorted(best_per_dim.values(), key=lambda c: c["_abs"], reverse=True)[:3]
    for c in ranked:
        c.pop("_abs")

    return ranked


if __name__ == "__main__":
    df = pd.read_csv("data_utils/raw_data/officer_twins_live.csv")
    sample = df.iloc[0]

    print(f"=== SHAP top_3 for officer {sample['officer_id']} (day {sample['day']}) ===")
    top3 = get_top_3_shap(sample)
    for c in top3:
        print(f"  {c['dim_code']} ({c['dim_label']}): raw={c['raw_value']:.4f}  shap={c['shap_contribution_pts']} pts")