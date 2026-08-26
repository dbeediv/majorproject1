import pandas as pd
import sys
sys.path.insert(0, ".")

from contracts import OfficerTwin
from models.agent1.cold_start import (
    compute_days_of_history, compute_twin_confidence,
    determine_normalization_baseline, apply_cold_start_cap
)
from models.agent1.build_officer_twin import build_twin_for_officer_day

DIMENSIONS = ["d1_approval_rate", "d2_garage_concentration", "d3_threshold_proximity",
              "d4_surveyor_diversity", "d5_turnaround_variance", "d6_after_hours_pct"]


def test_days_of_history_capping():
    """days_of_history must never exceed 60, per the schema's le=60 rule."""
    assert compute_days_of_history(1) == 1
    assert compute_days_of_history(60) == 60
    assert compute_days_of_history(61) == 60   # capped
    assert compute_days_of_history(200) == 60  # still capped
    print("PASS: days_of_history capping")


def test_twin_confidence_reaches_exactly_one_at_day_61():
    """
    Regression test for the bug found and fixed this session: confidence
    must reach EXACTLY 1.0 on day 61, the same day baseline becomes
    'personal' -- not 0.9836 (the capped-days_of_history bug).
    """
    assert compute_twin_confidence(61) == 1.0
    assert compute_twin_confidence(90) == 1.0
    assert compute_twin_confidence(60) < 1.0  # day 60 should NOT be full confidence yet
    print("PASS: twin_confidence reaches exactly 1.0 at day 61 (bug fix verified)")


def test_baseline_transitions_at_correct_days():
    """Baseline must switch exactly at day 31 and day 61, nowhere else."""
    assert determine_normalization_baseline(30) == "population"
    assert determine_normalization_baseline(31) == "blended"
    assert determine_normalization_baseline(60) == "blended"
    assert determine_normalization_baseline(61) == "personal"
    print("PASS: baseline transitions occur at exactly the right days")


def test_confidence_and_baseline_are_consistent():
    """
    The core design guarantee from Fix A-1: confidence hits 1.0 on the
    SAME day baseline becomes 'personal'. No one-day mismatch window.
    """
    for day in [61, 62, 100]:
        conf = compute_twin_confidence(day)
        baseline = determine_normalization_baseline(day)
        assert conf == 1.0 and baseline == "personal", \
            f"Day {day}: confidence={conf}, baseline={baseline} -- should both be 'fully mature'"
    print("PASS: twin_confidence and normalization_baseline are always consistent")


def test_score_cap_during_cold_start():
    """A score must be capped at 70 unless baseline is 'personal'."""
    assert apply_cold_start_cap(95.0, "population") == 70.0
    assert apply_cold_start_cap(95.0, "blended") == 70.0
    assert apply_cold_start_cap(95.0, "personal") == 95.0
    assert apply_cold_start_cap(50.0, "population") == 50.0  # scores below cap pass through unchanged
    print("PASS: score cap enforced correctly during cold start")


def test_officer_twin_schema_validation_on_real_data():
    """
    The real integration test: build actual OfficerTwin records from real
    Stage 4/5 output and confirm every single one passes strict Pydantic
    validation (extra='forbid', all field constraints).
    """
    df = pd.read_csv("data_utils/raw_data/synthetic_officers_scored.csv")
    sample_officer_id = df["officer_id"].iloc[0]
    officer_data = df[df["officer_id"] == sample_officer_id].sort_values("day")

    built_count = 0
    for _, row in officer_data.iterrows():
        z_scores = {f"{dim}_z": row[f"{dim}_z"] for dim in DIMENSIONS}
        twin = build_twin_for_officer_day(row, z_scores, int(row["day"]))
        assert isinstance(twin, OfficerTwin)  # would have raised already if invalid
        built_count += 1

    assert built_count == 60, f"Expected 60 valid twin records, got {built_count}"
    print(f"PASS: all {built_count} OfficerTwin records for a real officer passed schema validation")


def test_officer_id_format():
    """officer_id must always match OFF-XXXXXX (6-digit hyphen format)."""
    df = pd.read_csv("data_utils/raw_data/synthetic_officers.csv")
    sample_id = df["officer_id"].iloc[0]
    import re
    assert re.match(r"^OFF-\d{6}$", sample_id), f"Bad officer_id format: {sample_id}"
    print("PASS: officer_id format is OFF-XXXXXX")


if __name__ == "__main__":
    print("Running Person A full pipeline test suite...\n")

    test_days_of_history_capping()
    test_twin_confidence_reaches_exactly_one_at_day_61()
    test_baseline_transitions_at_correct_days()
    test_confidence_and_baseline_are_consistent()
    test_score_cap_during_cold_start()
    test_officer_id_format()
    test_officer_twin_schema_validation_on_real_data()

    print("\nAll tests passed. Pipeline is ready for handoff to Person B.")