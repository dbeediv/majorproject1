import pandas as pd
from datetime import date, timedelta
import sys
sys.path.insert(0, ".")
from contracts import OfficerTwin
from models.agent1.cold_start import compute_days_of_history, compute_twin_confidence, determine_normalization_baseline


def build_twin_for_officer_day(officer_row: pd.Series, if_z_scores: dict, day: int) -> OfficerTwin:
    """
    Assembles one real, schema-valid OfficerTwin record for a specific
    officer on a specific day, combining:
    - raw D1-D6 values (from the synthetic generator)
    - population z-scores (from Stage 4's Isolation Forest z-score layer)
    - cold-start fields (from this stage)
    """
    days_of_history = compute_days_of_history(day)
    twin_confidence = compute_twin_confidence(day)
    normalization_baseline = determine_normalization_baseline(day)

    start_date = date(2026, 6, 1)
    twin_date = start_date + timedelta(days=day - 1)

    return OfficerTwin(
        officer_id=officer_row["officer_id"],
        twin_date=twin_date,
        d1_approval_rate=float(officer_row["d1_approval_rate"]),
        d1_approval_rate_z=float(if_z_scores["d1_approval_rate_z"]),
        d2_garage_concentration=float(officer_row["d2_garage_concentration"]),
        d2_garage_concentration_z=float(if_z_scores["d2_garage_concentration_z"]),
        d3_threshold_proximity=float(officer_row["d3_threshold_proximity"]),
        d3_threshold_proximity_z=float(if_z_scores["d3_threshold_proximity_z"]),
        d4_surveyor_diversity=float(officer_row["d4_surveyor_diversity"]),
        d4_surveyor_diversity_z=float(if_z_scores["d4_surveyor_diversity_z"]),
        d5_turnaround_variance=float(officer_row["d5_turnaround_variance"]),
        d5_turnaround_variance_z=float(if_z_scores["d5_turnaround_variance_z"]),
        d6_after_hours_pct=float(officer_row["d6_after_hours_pct"]),
        d6_after_hours_pct_z=float(if_z_scores["d6_after_hours_pct_z"]),
        twin_confidence=twin_confidence,
        days_of_history=days_of_history,
        normalization_baseline=normalization_baseline,
    )


if __name__ == "__main__":
    print("Loading scored data (from Stage 4)...")
    df = pd.read_csv("data_utils/raw_data/synthetic_officers_scored.csv")

    DIMENSIONS = ["d1_approval_rate", "d2_garage_concentration", "d3_threshold_proximity",
                  "d4_surveyor_diversity", "d5_turnaround_variance", "d6_after_hours_pct"]

    print("Building a real OfficerTwin batch (one officer, all 60 days)...")
    sample_officer_id = df[df["archetype"] == "gradually_colluding"]["officer_id"].iloc[0]
    officer_data = df[df["officer_id"] == sample_officer_id].sort_values("day")

    twins = []
    for _, row in officer_data.iterrows():
        z_scores = {f"{dim}_z": row[f"{dim}_z"] for dim in DIMENSIONS}
        twin = build_twin_for_officer_day(row, z_scores, int(row["day"]))
        twins.append(twin)

    print(f"\nBuilt {len(twins)} valid OfficerTwin records for {sample_officer_id}")
    print(f"\nSample record (day 1):")
    print(twins[0])
    print(f"\nSample record (day 35, should be 'blended'):")
    print(twins[34])
    print(f"\nSample record (day 60):")
    print(twins[59])

    # Save a real batch as the actual Stage 7 handoff artifact
    import json
    batch_json = [t.model_dump(mode="json") for t in twins]
    with open("data_utils/raw_data/officertwin_sample_batch.json", "w") as f:
        json.dump(batch_json, f, indent=2)
    print(f"\nSaved {len(twins)} records to officertwin_sample_batch.json")