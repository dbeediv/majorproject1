def compute_days_of_history_uncapped(day: int) -> int:
    """The true count of days of personal history -- used for confidence math."""
    return day


def compute_days_of_history(day: int) -> int:
    """
    The value actually EMITTED in the OfficerTwin record, capped at 60
    per the schema (le=60). Once baseline is 'personal', the exact count
    beyond 60 doesn't change behavior, so capping the stored field is a
    safe simplification -- see reference doc note on this exact point.
    """
    return min(day, 60)


def compute_twin_confidence(day: int) -> float:
    """
    twin_confidence = uncapped_days_of_history / 61, clamped [0.0, 1.0].
    IMPORTANT: uses the UNCAPPED day count, not the capped days_of_history
    field -- otherwise confidence can never reach 1.0 (60/61 = 0.9836 forever).
    """
    raw = day / 61
    return max(0.0, min(1.0, raw))


def determine_normalization_baseline(day: int) -> str:
    if day <= 30:
        return "population"
    elif day <= 60:
        return "blended"
    else:
        return "personal"


def apply_cold_start_cap(score: float, normalization_baseline: str) -> float:
    """
    Caps the final risk score at 70 during cold start (population or
    blended baseline), preventing new officers from being escalated to
    severe response tiers (80+) before the twin has matured.
    """
    if normalization_baseline != "personal":
        return min(score, 70.0)
    return score


if __name__ == "__main__":
    for day in [1, 15, 30, 31, 45, 60, 61, 90]:
        doh = compute_days_of_history(day)
        conf = compute_twin_confidence(day)
        baseline = determine_normalization_baseline(day)
        print(f"day={day:3d}  days_of_history={doh:2d}  twin_confidence={conf:.4f}  baseline={baseline}")

    print("\n--- Score cap test ---")
    # A spuriously high score of 95 during cold start should be capped at 70
    capped = apply_cold_start_cap(95.0, "population")
    print(f"Score 95 during 'population' baseline -> capped to: {capped}")
    assert capped == 70.0, f"Expected 70.0, got {capped}"

    # Same score once baseline is personal should pass through unchanged
    uncapped = apply_cold_start_cap(95.0, "personal")
    print(f"Score 95 during 'personal' baseline -> stays: {uncapped}")
    assert uncapped == 95.0, f"Expected 95.0, got {uncapped}"

    print("Score cap tests passed.")