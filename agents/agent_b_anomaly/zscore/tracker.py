from typing import Dict, List, Tuple
from datetime import date


class PersonalBaselineStore:
    """
    Maintains historical rolling baselines (60-day window) per officer.
    Ensures today's measurement is scored strictly against PREVIOUS history 
    before being appended to avoid baseline contamination.
    """
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        # Structure: officer_id -> {"d1": [previous_history_values], ...}
        self.history: Dict[str, Dict[str, List[float]]] = {}

    def get_baseline_stats(
        self, officer_id: str, default_dims: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculates mean and std strictly from historical days prior to today[cite: 4]."""
        if officer_id not in self.history or not self.history[officer_id]["d1"]:
            # Cold-start fallback to incoming measurement with unit variance
            means = {k: float(v) for k, v in default_dims.items()}
            stds = {k: 1.0 for k in default_dims.keys()}
            return means, stds

        means, stds = {}, {}
        for dim, hist in self.history[officer_id].items():
            if len(hist) == 0:
                means[dim] = float(default_dims.get(dim, 0.0))
                stds[dim] = 1.0
            else:
                mu = sum(hist) / len(hist)
                variance = sum((x - mu) ** 2 for x in hist) / len(hist)
                sigma = max(variance ** 0.5, self.epsilon)
                means[dim] = round(mu, 4)
                stds[dim] = round(sigma, 4)

        return means, stds

    def update_history(self, officer_id: str, raw_dims: Dict[str, float]):
        """Appends today's measurements to rolling history AFTER anomaly scoring completes[cite: 4]."""
        if officer_id not in self.history:
            self.history[officer_id] = {f"d{i}": [] for i in range(1, 7)}

        for dim, val in raw_dims.items():
            hist = self.history[officer_id][dim]
            hist.append(float(val))
            if len(hist) > 60:
                hist.pop(0)


class RegionalSurgeController:
    """
    Monitors global anomaly concentration by regional zone[cite: 1, 3, 4].
    If flagged_count / total >= 30%, activates a 1.20x threshold inflation for 30 days[cite: 1, 3, 4].
    """
    def __init__(self):
        self.active_surges: Dict[str, date] = {}

    def update_and_check(
        self, region_id: str, current_date: date, flagged_count: int, total_count: int
    ) -> bool:
        if total_count > 0 and (flagged_count / total_count) >= 0.30:
            self.active_surges[region_id] = current_date
            return True

        if region_id in self.active_surges:
            activation_date = self.active_surges[region_id]
            days_active = (current_date - activation_date).days
            if days_active <= 30:
                return True
            else:
                del self.active_surges[region_id]

        return False


class PersonalZScoreTracker:
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def compute_z_scores(
        self,
        raw_dims: Dict[str, float],
        personal_mean: Dict[str, float],
        personal_std: Dict[str, float],
    ) -> Dict[str, float]:
        z_scores = {}
        for dim_key in ["d1", "d2", "d3", "d4", "d5", "d6"]:
            val = float(raw_dims.get(dim_key, 0.0))
            mu = float(personal_mean.get(dim_key, 0.0))
            sigma = max(float(personal_std.get(dim_key, self.epsilon)), self.epsilon)
            z_scores[dim_key] = round(float((val - mu) / sigma), 4)
        return z_scores


class SustainedStateTracker:
    def __init__(self):
        self.history: Dict[str, List[bool]] = {}

    def update(self, officer_id: str, is_daily_anomaly: bool) -> int:
        if officer_id not in self.history:
            self.history[officer_id] = []
        window = self.history[officer_id]
        window.append(bool(is_daily_anomaly))
        if len(window) > 7:
            window.pop(0)
        return sum(window)