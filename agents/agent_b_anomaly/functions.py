import os
from datetime import date
from typing import Dict, Optional, Any, List, Set
import numpy as np
import torch

from contracts import AnomalyFlag, OfficerTwin
from agents.agent_b_anomaly.models.autoencoder import (
    SymmetricAutoencoder,
    AutoencoderScorer,
)
from agents.agent_b_anomaly.zscore.tracker import (
    PersonalZScoreTracker,
    SustainedStateTracker,
    PersonalBaselineStore,
    RegionalSurgeController,
)
from agents.agent_b_anomaly.graph.ring_detector import CollusionRingDetector


def get_latest_anomaly_flag(officer_id: str) -> AnomalyFlag:
    return AnomalyFlag(
        officer_id=officer_id if officer_id.startswith("OFF-") else f"OFF-{officer_id}",
        flag_date=date.today(),
        per_dimension_zscore={"d1": 0.2, "d2": 3.4, "d3": 3.1, "d4": 0.1, "d5": 3.2, "d6": 0.0},
        autoencoder_reconstruction_error=0.18,
        sustained_days=5,
        network_ring_id="RING-000001",
        hard_flag=True,
    )


def get_ring_graph(officer_id: str, ring_detector: Optional[CollusionRingDetector] = None) -> Dict[str, Any]:
    if ring_detector is not None:
        return ring_detector.get_real_ring_graph(officer_id)
    clean_id = officer_id if officer_id.startswith("OFF-") else f"OFF-{officer_id}"
    return {
        "officer_id": clean_id,
        "network_ring_id": "RING-000001",
        "nodes": [clean_id, "GAR-000042", "OFF-000089"],
        "edges": [[clean_id, "GAR-000042"], ["OFF-000089", "GAR-000042"]],
        "overlap_score": 0.75,
    }


class Agent2Pipeline:
    def __init__(
        self,
        ae_scorer: AutoencoderScorer,
        ring_detector: CollusionRingDetector,
        ae_cutoff_percentile: float = 0.75,
    ):
        self.ae_scorer = ae_scorer
        self.ring_detector = ring_detector
        self.z_tracker = PersonalZScoreTracker()
        self.baseline_store = PersonalBaselineStore()
        self.sustained_tracker = SustainedStateTracker()
        self.surge_controller = RegionalSurgeController()
        self.ae_cutoff_percentile = ae_cutoff_percentile
        self._active_flagged_pool: Set[str] = set()

    @classmethod
    def from_pretrained(
        cls,
        weights_path: str = "agents/agent_b_anomaly/weights/autoencoder.pt",
        baseline_errors_path: str = "agents/agent_b_anomaly/weights/baseline_errors.npy",
        jaccard_threshold: float = 0.60,
    ) -> "Agent2Pipeline":
        model = SymmetricAutoencoder()
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()

        if os.path.exists(baseline_errors_path):
            baseline_errors = np.load(baseline_errors_path)
        else:
            baseline_errors = np.array([], dtype=np.float32)

        scorer = AutoencoderScorer(model=model, baseline_errors=baseline_errors)
        ring_detector = CollusionRingDetector(jaccard_threshold=jaccard_threshold)
        return cls(ae_scorer=scorer, ring_detector=ring_detector)

    def ingest_claim(self, officer_id: str, garage_id: str):
        self.ring_detector.add_claim(officer_id, garage_id)

    def process_officer_day(
        self,
        twin: OfficerTwin,
        personal_mean: Optional[Dict[str, float]] = None,
        personal_std: Optional[Dict[str, float]] = None,
        region_id: str = "REG-01",
        regional_flagged_count: int = 0,
        regional_total_count: int = 100,
    ) -> AnomalyFlag:
        raw_dims = {
            "d1": twin.d1_approval_rate,
            "d2": twin.d2_garage_concentration,
            "d3": twin.d3_threshold_proximity,
            "d4": twin.d4_surveyor_diversity,
            "d5": twin.d5_turnaround_variance,
            "d6": twin.d6_after_hours_pct,
        }

        z_vector = [
            twin.d1_approval_rate_z,
            twin.d2_garage_concentration_z,
            twin.d3_threshold_proximity_z,
            twin.d4_surveyor_diversity_z,
            twin.d5_turnaround_variance_z,
            twin.d6_after_hours_pct_z,
        ]

        # 1. Fetch uncontaminated baseline stats from prior history
        if personal_mean is None or personal_std is None:
            personal_mean, personal_std = self.baseline_store.get_baseline_stats(
                twin.officer_id, raw_dims
            )

        # 2. Compute personal Z-scores against prior baseline[cite: 4]
        z_scores = self.z_tracker.compute_z_scores(raw_dims, personal_mean, personal_std)
        extreme_dims = sum(1 for v in z_scores.values() if v > 3.0)

        # 3. Append today's values to history AFTER scoring to avoid leakage[cite: 4]
        self.baseline_store.update_history(twin.officer_id, raw_dims)

        ae_error = self.ae_scorer.score(z_vector)

        # 4. Regional macro surge threshold adjustment[cite: 1, 4]
        is_surge = self.surge_controller.update_and_check(
            region_id, twin.twin_date, regional_flagged_count, regional_total_count
        )
        effective_cutoff = (
            min(1.0, self.ae_cutoff_percentile * 1.20)
            if is_surge
            else self.ae_cutoff_percentile
        )

        is_daily_anomaly = (extreme_dims >= 3) or (ae_error > effective_cutoff)
        sustained = self.sustained_tracker.update(twin.officer_id, is_daily_anomaly)
        hard_flag = (sustained >= 5) and is_daily_anomaly

        if is_daily_anomaly:
            self._active_flagged_pool.add(twin.officer_id)
            self.ring_detector.evaluate_rings(list(self._active_flagged_pool))
        elif twin.officer_id in self._active_flagged_pool:
            self._active_flagged_pool.remove(twin.officer_id)

        ring_id = self.ring_detector.get_ring_id(twin.officer_id)

        return AnomalyFlag(
            officer_id=twin.officer_id,
            flag_date=twin.twin_date,
            per_dimension_zscore=z_scores,
            autoencoder_reconstruction_error=round(float(ae_error), 4),
            sustained_days=sustained,
            network_ring_id=ring_id,
            hard_flag=hard_flag,
        )