from contracts import OfficerTwin
from datetime import date

t = OfficerTwin(
    officer_id="OFF-000001",
    twin_date=date.today(),
    d1_approval_rate=0.8, d1_approval_rate_z=0.5,
    d2_garage_concentration=0.3, d2_garage_concentration_z=0.2,
    d3_threshold_proximity=0.1, d3_threshold_proximity_z=0.1,
    d4_surveyor_diversity=10.0, d4_surveyor_diversity_z=0.0,
    d5_turnaround_variance=2.5, d5_turnaround_variance_z=0.3,
    d6_after_hours_pct=0.05, d6_after_hours_pct_z=0.1,
    twin_confidence=0.5, days_of_history=30,
    normalization_baseline="blended"
)
print("Schema works:", t)