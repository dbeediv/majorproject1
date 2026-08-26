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


import pandas as pd
import numpy as np


def compute_blended_zscore(raw_value: float, day: int, dimension: str,
                             population_mean: float, population_std: float,
                             officer_history: pd.Series) -> float:
    """
    Computes a z-score using the appropriate baseline for this day,
    per the cold-start blending rule:
    - days 1-30: population baseline only
    - days 31-60: 50/50 blend of population and personal-so-far
    - day 61+: fully personal baseline

    officer_history: this officer's own values for this dimension,
    for all days up to (but not including) the current day -- i.e.
    "personal history so far," which grows as day increases.
    """
    baseline = determine_normalization_baseline(day)

    if baseline == "population":
        mean, std = population_mean, population_std
    elif baseline == "personal":
        mean, std = officer_history.mean(), officer_history.std()
    else:  # blended
        personal_mean = officer_history.mean() if len(officer_history) > 0 else population_mean
        personal_std = officer_history.std() if len(officer_history) > 1 else population_std
        mean = 0.5 * population_mean + 0.5 * personal_mean
        std = 0.5 * population_std + 0.5 * personal_std

    return (raw_value - mean) / (std + 1e-9)


def test_transition_smoothness():
    """
    Tests days 29-32 and days 59-62 for one real officer, checking that
    the z-score doesn't jump drastically just because the baseline
    formula switched -- a real behavioral shift should look gradual,
    not a computation artifact.
    """
    df = pd.read_csv("data_utils/raw_data/synthetic_officers.csv")

    # Use a clean officer -- their D2 should be genuinely stable, so
    # any big jump we see is a formula bug, not real behavior changing
    officer_id = df[df["archetype"] == "clean"]["officer_id"].iloc[0]
    officer_df = df[df["officer_id"] == officer_id].sort_values("day").reset_index(drop=True)

    population_mean = df["d2_garage_concentration"].mean()
    population_std = df["d2_garage_concentration"].std()

    print(f"Testing officer {officer_id} (clean) — D2 z-score across baseline transitions\n")

    for test_range, label in [(range(28, 33), "Day 30/31 transition"), (range(58, 63), "Day 60/61 transition")]:
        print(f"--- {label} ---")
        for day in test_range:
            if day > 60:
                # simulate "beyond day 60" using full 60-day history as personal
                history = officer_df["d2_garage_concentration"]
            else:
                history = officer_df[officer_df["day"] < day]["d2_garage_concentration"]

            raw_value = officer_df[officer_df["day"] == min(day, 60)]["d2_garage_concentration"].iloc[0]
            z = compute_blended_zscore(raw_value, day, "d2_garage_concentration",
                                         population_mean, population_std, history)
            baseline = determine_normalization_baseline(day)
            print(f"  day={day:2d}  baseline={baseline:10s}  z-score={z:+.4f}")
        print()


def diagnose_transition_jump():
    """
    Checks WHY the day-60/61 jump happens: is it purely the expected
    consequence of shifting from a 50% population weight to 100% personal
    weight (which the design explicitly specifies as a step, not a gradual
    ramp), or is there an additional bug inflating it further?
    """
    df = pd.read_csv("data_utils/raw_data/synthetic_officers.csv")
    officer_id = df[df["archetype"] == "clean"]["officer_id"].iloc[0]
    officer_df = df[df["officer_id"] == officer_id].sort_values("day").reset_index(drop=True)

    population_mean = df["d2_garage_concentration"].mean()
    population_std = df["d2_garage_concentration"].std()
    personal_mean = officer_df["d2_garage_concentration"].mean()  # full 60-day personal average
    personal_std = officer_df["d2_garage_concentration"].std()

    print(f"Population D2: mean={population_mean:.4f}, std={population_std:.4f}")
    print(f"This officer's personal D2 (full 60 days): mean={personal_mean:.4f}, std={personal_std:.4f}")
    print(f"Difference in means: {abs(population_mean - personal_mean):.4f}")
    print(f"Difference in stds: {abs(population_std - personal_std):.4f}")
    print()
    print("If personal and population stats are close, the blend->personal jump")
    print("should be small. If they differ a lot, a jump is an EXPECTED consequence")
    print("of the design's step-function blend (population -> 50/50 -> personal),")
    print("not a bug -- the design specifies discrete stages, not a smooth ramp.")


if __name__ == "__main__":
    for day in [1, 15, 30, 31, 45, 60, 61, 90]:
        doh = compute_days_of_history(day)
        conf = compute_twin_confidence(day)
        baseline = determine_normalization_baseline(day)
        print(f"day={day:3d}  days_of_history={doh:2d}  twin_confidence={conf:.4f}  baseline={baseline}")

    print("\n--- Score cap test ---")
    capped = apply_cold_start_cap(95.0, "population")
    print(f"Score 95 during 'population' baseline -> capped to: {capped}")
    assert capped == 70.0
    uncapped = apply_cold_start_cap(95.0, "personal")
    print(f"Score 95 during 'personal' baseline -> stays: {uncapped}")
    assert uncapped == 95.0
    print("Score cap tests passed.")

    print("\n--- Transition smoothness test ---")
    test_transition_smoothness()

    print("\n--- Diagnosing the day 60/61 jump ---")
    diagnose_transition_jump()
