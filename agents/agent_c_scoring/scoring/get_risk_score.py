# get_risk_score.py
# Person C - Stage 2. Replaces mock_risk_score.get_mock_risk_score() with the
# real pipeline: model probability -> decay -> band -> shap -> RiskScore.
#
# TODO: twin_confidence_at_scoring is stubbed below. officer_twins_live.csv
# does not contain this field yet - it is only in Person A's updated CSV
# (blocked from git, >100MB). Replace the stub once she sends it.

import sys
from pathlib import Path

import joblib
import pandas as pd
from datetime import date

sys.path.append(str(Path(__file__).resolve().parents[3]))
from contracts import RiskScore  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from shap_explain import get_top_3_shap, FEATURE_COLS  # noqa: E402
from tier_rules import compute_score_band  # noqa: E402
from decay import update_last_clean_date, clean_days, apply_decay_with_band  # noqa: E402

MODEL_PATH = Path("agents/agent_c_scoring/results/live_model.pkl")

_model = None


def _load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def get_risk_score(feature_row: pd.Series, hard_flag_today: bool, today: date = None) -> RiskScore:
    """
    feature_row: one officer/day row from officer_twins_live.csv (must
        contain FEATURE_COLS + officer_id).
    hard_flag_today: B's real-time anomaly signal for this officer/day -
        feeds decay.update_last_clean_date() per Fix D-5's mandatory ordering.
    """
    if today is None:
        today = date.today()

    officer_id = feature_row["officer_id"]
    model = _load_model()

    X = feature_row[FEATURE_COLS].to_frame().T.apply(pd.to_numeric, errors="coerce")
    proba = model.predict_proba(X)[0][1]
    raw_score = round(proba * 100)
    raw_score = max(0, min(100, raw_score))
    band = compute_score_band(raw_score)

    update_last_clean_date(officer_id, hard_flag_today=hard_flag_today, today=today)
    days_clean = clean_days(officer_id, today)
    decayed_score, decayed_band = apply_decay_with_band(raw_score, band, days_clean)
    decay_applied = decayed_score != raw_score

    top_3_shap = get_top_3_shap(feature_row)

    return RiskScore(
        officer_id=officer_id,
        score=decayed_score,
        score_band=decayed_band,
        top_3_shap=top_3_shap,
        score_date=today,
        decay_applied=decay_applied,
        financial_exposure_inr=None,
        twin_confidence_at_scoring=0.75,
    )


if __name__ == "__main__":
    df = pd.read_csv("data_utils/raw_data/officer_twins_live.csv")
    sample = df.iloc[0]
    rs = get_risk_score(sample, hard_flag_today=False)
    print(rs.model_dump_json(indent=2))
