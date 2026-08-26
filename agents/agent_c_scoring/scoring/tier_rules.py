# tier_rules.py
# Loads tier_config.yaml and exposes compute_score_band() — the function
# C-2's apply_decay_with_band() wrapper calls to recompute score_band after
# decay. Keeping this separate from the YAML means D's weekly job and C's
# live scoring both call the SAME function, so bands can never drift apart.

import sys
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "tier_config.yaml"

# Must match RiskScore.score_band Literal exactly — contracts/__init__.py
# is the actual source of truth, this is just a local sanity list.
VALID_BANDS = {"clean", "watch", "suspected", "high_risk", "confirmed"}


def load_tier_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    band_names = {b["name"] for b in config["bands"]}
    if band_names != VALID_BANDS:
        raise ValueError(
            f"tier_config.yaml band names {band_names} do not match "
            f"the contract's score_band values {VALID_BANDS}"
        )

    # sanity check: bands should be contiguous 0-100 with no gaps/overlaps
    sorted_bands = sorted(config["bands"], key=lambda b: b["min_score"])
    expected_start = 0
    for b in sorted_bands:
        if b["min_score"] != expected_start:
            raise ValueError(
                f"Gap or overlap in tier_config.yaml near band '{b['name']}': "
                f"expected min_score={expected_start}, got {b['min_score']}"
            )
        expected_start = b["max_score"] + 1
    if expected_start != 101:
        raise ValueError(f"Bands don't cover the full 0-100 range, end at {expected_start - 1}")

    return config


_CONFIG = load_tier_config()
_SORTED_BANDS = sorted(_CONFIG["bands"], key=lambda b: b["min_score"])


def compute_score_band(score: int) -> str:
    """Pure function: score (0-100) -> band name. This is what C-2's
    apply_decay_with_band() wrapper calls after decaying a score, and what
    live scoring calls when producing a fresh RiskScore."""
    if not (0 <= score <= 100):
        raise ValueError(f"score must be 0-100, got {score}")
    for b in _SORTED_BANDS:
        if b["min_score"] <= score <= b["max_score"]:
            return b["name"]
    raise ValueError(f"No band found for score={score} — check tier_config.yaml")


def alert_tier_for_band(band: str) -> int | None:
    """Returns the AlertPayload.tier (2-5) for a given band, or None if
    the band (clean/watch) never generates an alert."""
    return _CONFIG["alert_tiers"].get(band)


def is_decay_paused(band: str) -> bool:
    """high_risk and confirmed never auto-decay — human review only."""
    de_esc = _CONFIG["de_escalation"].get(band, {})
    return de_esc.get("decay_paused", False)


if __name__ == "__main__":
    print("=== tier_config.yaml loaded and validated ===\n")

    test_scores = [0, 20, 39, 40, 55, 59, 60, 75, 79, 80, 90, 94, 95, 100]
    print("Score -> Band -> Alert Tier -> Decay Paused?")
    for s in test_scores:
        band = compute_score_band(s)
        tier = alert_tier_for_band(band)
        paused = is_decay_paused(band)
        print(f"  {s:3d} -> {band:10s} -> tier={tier}  decay_paused={paused}")

    print("\nAll bands validated: contiguous 0-100, names match contract, no gaps.")