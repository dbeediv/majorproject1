# MOCK STUB — Stage 1. Returns contract-compliant fake RiskScore/AlertPayload
# objects so D can build dashboard + API routes against real schema shapes
# without waiting for the real Agent 3 model (blocked on A's _z fields).
# Swap get_mock_risk_score() for the real get_risk_score() in Stage 2 —
# D's route signature does not change.

import random
from datetime import date, datetime, timezone

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))
from contracts import RiskScore, ShapContribution, AlertPayload  # noqa: E402

DIM_LABELS = {
    "d1": "Approval Rate",
    "d2": "Garage Concentration",
    "d3": "Threshold Proximity",
    "d4": "Surveyor Diversity",
    "d5": "Turnaround Variance",
    "d6": "After-Hours Activity",
}

SCORE_BANDS = ["clean", "watch", "suspected", "high_risk", "confirmed"]


def _band_for_score(score: int) -> str:
    if score <= 39:
        return "clean"
    if score <= 59:
        return "watch"
    if score <= 79:
        return "suspected"
    if score <= 94:
        return "high_risk"
    return "confirmed"


def get_mock_risk_score(officer_id: str, seed: int | None = None) -> RiskScore:
    """Returns one fake but schema-valid RiskScore for a given officer_id.
    D calls this in place of the real Agent 3 output during Stage 1."""
    rng = random.Random(seed)

    score = rng.randint(0, 100)
    band = _band_for_score(score)

    # exactly 3 ShapContribution objects — schema enforces min/max length 3
    chosen_dims = rng.sample(list(DIM_LABELS.keys()), 3)
    top_3_shap = [
        ShapContribution(
            dim_code=dim,
            dim_label=DIM_LABELS[dim],
            raw_value=round(rng.uniform(0, 1), 4),
            shap_contribution_pts=round(rng.uniform(-15, 15), 2),
        )
        for dim in chosen_dims
    ]

    return RiskScore(
        officer_id=officer_id,
        score=score,
        score_band=band,
        top_3_shap=top_3_shap,
        score_date=date.today(),
        decay_applied=rng.choice([True, False]),
        financial_exposure_inr=round(rng.uniform(0, 500000), 2) if band in ("high_risk", "confirmed") else None,
        twin_confidence_at_scoring=round(rng.uniform(0.3, 1.0), 2),
    )


def get_mock_alert_payload(officer_id: str | None = None, officer_ids: list[str] | None = None,
                            seed: int | None = None) -> AlertPayload:
    """Returns one fake but schema-valid AlertPayload. Pass either officer_id
    (single-officer alert) OR officer_ids (joint ring case), not both."""
    rng = random.Random(seed)
    return AlertPayload(
        officer_id=officer_id,
        officer_ids=officer_ids,
        tier=rng.randint(2, 5),
        recipients=["compliance@guardclaim.example"],
        channel=["dashboard", "email"],
        generated_at=datetime.now(timezone.utc),
        evidence_pdf_url=None,
    )


if __name__ == "__main__":
    # Quick self-check: generate a few mock officers, verify they validate
    # against the real contract (StrictModel raises on any schema mismatch).
    test_officer_ids = ["OFF-000001", "OFF-000042", "OFF-009999"]

    print("=== Mock RiskScore objects (contract-validated) ===")
    for i, oid in enumerate(test_officer_ids):
        rs = get_mock_risk_score(oid, seed=i)
        print(f"\n{oid}:")
        print(f"  score={rs.score}  band={rs.score_band}  decay_applied={rs.decay_applied}")
        print(f"  twin_confidence_at_scoring={rs.twin_confidence_at_scoring}")
        print(f"  financial_exposure_inr={rs.financial_exposure_inr}")
        print(f"  top_3_shap:")
        for shap in rs.top_3_shap:
            print(f"    {shap.dim_code} ({shap.dim_label}): {shap.shap_contribution_pts} pts")

    print("\n=== Mock AlertPayload — single officer ===")
    alert1 = get_mock_alert_payload(officer_id="OFF-000001", seed=1)
    print(alert1.model_dump_json(indent=2))

    print("\n=== Mock AlertPayload — joint ring case ===")
    alert2 = get_mock_alert_payload(officer_ids=["OFF-000001", "OFF-000042"], seed=2)
    print(alert2.model_dump_json(indent=2))

    print("\nAll objects validated successfully against contracts/__init__.py")