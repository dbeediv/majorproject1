import os
from datetime import date
import pandas as pd
import numpy as np
import pytest

from contracts import AnomalyFlag, OfficerTwin
from agents.agent_b_anomaly.graph.ring_detector import CollusionRingDetector
from agents.agent_b_anomaly.models.autoencoder import (
    AutoencoderScorer,
    SymmetricAutoencoder,
)
from agents.agent_b_anomaly.zscore.tracker import (
    PersonalZScoreTracker,
    SustainedStateTracker,
    RegionalSurgeController,
)
from agents.agent_b_anomaly.functions import Agent2Pipeline, get_ring_graph


# 1. Pydantic Regex & Schema Conformance
def test_anomaly_flag_schema_and_ring_regex():
    flag = AnomalyFlag(
        officer_id="OFF-000001",
        flag_date=date.today(),
        per_dimension_zscore={"d1": 0.5, "d2": 3.2, "d3": 3.1, "d4": 0.0, "d5": 3.5, "d6": 0.0},
        autoencoder_reconstruction_error=0.1850,
        sustained_days=5,
        network_ring_id="RING-000042",
        hard_flag=True,
    )
    assert flag.network_ring_id == "RING-000042"
    assert flag.hard_flag is True

    flag_no_ring = AnomalyFlag(
        officer_id="OFF-000002",
        flag_date=date.today(),
        per_dimension_zscore={"d1": 0.0, "d2": 0.0, "d3": 0.0, "d4": 0.0, "d5": 0.0, "d6": 0.0},
        autoencoder_reconstruction_error=0.0200,
        sustained_days=0,
        network_ring_id=None,
        hard_flag=False,
    )
    assert flag_no_ring.network_ring_id is None

    with pytest.raises(Exception):
        AnomalyFlag(
            officer_id="OFF-000003",
            flag_date=date.today(),
            per_dimension_zscore={"d1": 0.0, "d2": 0.0, "d3": 0.0, "d4": 0.0, "d5": 0.0, "d6": 0.0},
            autoencoder_reconstruction_error=0.10,
            sustained_days=1,
            network_ring_id="RING-42",
            hard_flag=False,
        )


# 2. Personal Rolling Z-Score Computation
def test_personal_zscore_calculation():
    tracker = PersonalZScoreTracker()
    raw = {"d1": 0.85, "d2": 0.70, "d3": 0.45, "d4": 12.0, "d5": 4.50, "d6": 0.30}
    mu = {"d1": 0.75, "d2": 0.20, "d3": 0.10, "d4": 12.0, "d5": 1.50, "d6": 0.05}
    std = {"d1": 0.05, "d2": 0.10, "d3": 0.05, "d4": 2.00, "d5": 1.00, "d6": 0.05}

    z = tracker.compute_z_scores(raw, mu, std)
    assert pytest.approx(z["d1"], rel=1e-3) == 2.0
    assert pytest.approx(z["d2"], rel=1e-3) == 5.0
    assert pytest.approx(z["d3"], rel=1e-3) == 7.0
    assert pytest.approx(z["d4"], rel=1e-3) == 0.0
    assert pytest.approx(z["d5"], rel=1e-3) == 3.0
    assert pytest.approx(z["d6"], rel=1e-3) == 5.0


# 3. Sustained Anomaly Rule
def test_sustained_state_tracker_5_of_7_rule():
    tracker = SustainedStateTracker()
    officer_id = "OFF-000042"
    for day in range(1, 5):
        assert tracker.update(officer_id, is_daily_anomaly=True) == day
    assert tracker.update(officer_id, is_daily_anomaly=True) == 5
    assert tracker.update(officer_id, is_daily_anomaly=False) == 5
    assert tracker.update(officer_id, is_daily_anomaly=False) == 5
    assert tracker.update(officer_id, is_daily_anomaly=False) == 4


# 4. Top-10 Garage Bipartite Collusion Ring Test
def test_top10_garage_3_officer_ring():
    detector = CollusionRingDetector(jaccard_threshold=0.60)

    shared_garages = [f"GAR-00000{i}" for i in range(1, 9)]
    for g in shared_garages:
        detector.add_claim("OFF-000001", g)
        detector.add_claim("OFF-000002", g)
        detector.add_claim("OFF-000003", g)

    detector.add_claim("OFF-000001", "GAR-000011")
    detector.add_claim("OFF-000001", "GAR-000012")
    detector.add_claim("OFF-000002", "GAR-000013")
    detector.add_claim("OFF-000002", "GAR-000014")

    detector.evaluate_rings(["OFF-000001", "OFF-000002", "OFF-000003"])

    r1 = detector.get_ring_id("OFF-000001")
    r2 = detector.get_ring_id("OFF-000002")
    r3 = detector.get_ring_id("OFF-000003")

    assert r1 == "RING-000001"
    assert r2 == "RING-000001"
    assert r3 == "RING-000001"

    graph_payload = get_ring_graph("OFF-000001", ring_detector=detector)
    assert graph_payload["network_ring_id"] == "RING-000001"
    assert "OFF-000001" in graph_payload["nodes"]


# 5. Multi-Officer Pipeline Ring Evaluation
def test_pipeline_multi_officer_ring_detection():
    pipeline = Agent2Pipeline.from_pretrained()

    officers = ["OFF-000101", "OFF-000102", "OFF-000103"]
    shared_garages = [f"GAR-00000{i}" for i in range(1, 9)]

    for off in officers:
        for gar in shared_garages:
            pipeline.ingest_claim(off, gar)

    for day in range(5):
        for off in officers:
            twin = OfficerTwin(
                officer_id=off,
                twin_date=date.today(),
                d1_approval_rate=0.98,
                d1_approval_rate_z=4.5,
                d2_garage_concentration=0.90,
                d2_garage_concentration_z=5.0,
                d3_threshold_proximity=0.85,
                d3_threshold_proximity_z=4.8,
                d4_surveyor_diversity=1.0,
                d4_surveyor_diversity_z=-3.5,
                d5_turnaround_variance=0.1,
                d5_turnaround_variance_z=-3.0,
                d6_after_hours_pct=0.80,
                d6_after_hours_pct_z=6.0,
                twin_confidence=1.0,
                days_of_history=60,
                normalization_baseline="personal",
            )
            personal_mu = {f"d{i}": 0.1 for i in range(1, 7)}
            personal_std = {f"d{i}": 0.05 for i in range(1, 7)}
            flag = pipeline.process_officer_day(
                twin=twin, personal_mean=personal_mu, personal_std=personal_std
            )

    r1 = pipeline.ring_detector.get_ring_id("OFF-000101")
    r2 = pipeline.ring_detector.get_ring_id("OFF-000102")
    r3 = pipeline.ring_detector.get_ring_id("OFF-000103")

    assert flag.hard_flag is True
    assert r1 == "RING-000001"
    assert r2 == "RING-000001"
    assert r3 == "RING-000001"


# 6. Live Integration Test against Real CSV Records
def test_real_dataset_a_to_b_integration():
    csv_path = "data_utils/raw_data/synthetic_officers.csv"
    if not os.path.exists(csv_path):
        pytest.skip("synthetic_officers.csv not found locally")

    df = pd.read_csv(csv_path)

    # 1. Enforce strict check on _z columns if present, otherwise calculate them dynamically
    raw_cols = [
        "d1_approval_rate",
        "d2_garage_concentration",
        "d3_threshold_proximity",
        "d4_surveyor_diversity",
        "d5_turnaround_variance",
        "d6_after_hours_pct",
    ]
    z_cols = [f"{col}_z" for col in raw_cols]

    sample_row = df.iloc[0]
    raw_off_id = sample_row["officer_id"]
    formatted_id = (
        f"OFF-{int(raw_off_id):06d}"
        if str(raw_off_id).isdigit()
        else str(raw_off_id)
    )

    # Use pre-computed _z columns or standard normalization fallback
    z_values = {}
    for raw, z in zip(raw_cols, z_cols):
        if z in df.columns:
            z_values[z] = float(sample_row[z])
        else:
            mean = df[raw].mean()
            std = max(df[raw].std(), 1e-6)
            z_values[z] = float((sample_row[raw] - mean) / std)

    twin = OfficerTwin(
        officer_id=formatted_id,
        twin_date=date.today(),
        d1_approval_rate=float(sample_row["d1_approval_rate"]),
        d1_approval_rate_z=z_values["d1_approval_rate_z"],
        d2_garage_concentration=float(sample_row["d2_garage_concentration"]),
        d2_garage_concentration_z=z_values["d2_garage_concentration_z"],
        d3_threshold_proximity=float(sample_row["d3_threshold_proximity"]),
        d3_threshold_proximity_z=z_values["d3_threshold_proximity_z"],
        d4_surveyor_diversity=float(sample_row["d4_surveyor_diversity"]),
        d4_surveyor_diversity_z=z_values["d4_surveyor_diversity_z"],
        d5_turnaround_variance=float(sample_row["d5_turnaround_variance"]),
        d5_turnaround_variance_z=z_values["d5_turnaround_variance_z"],
        d6_after_hours_pct=float(sample_row["d6_after_hours_pct"]),
        d6_after_hours_pct_z=z_values["d6_after_hours_pct_z"],
        twin_confidence=float(sample_row.get("twin_confidence", 1.0)),
        days_of_history=int(sample_row.get("days_of_history", 60)),
        normalization_baseline=str(sample_row.get("normalization_baseline", "personal")),
    )

    pipeline = Agent2Pipeline.from_pretrained()
    pipeline.ingest_claim(twin.officer_id, "GAR-000001")
    flag = pipeline.process_officer_day(twin=twin)

    assert isinstance(flag, AnomalyFlag)
    assert flag.officer_id == twin.officer_id
    assert 0.0 <= flag.autoencoder_reconstruction_error <= 1.0