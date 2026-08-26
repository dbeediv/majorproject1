import numpy as np
import pandas as pd
from datetime import date, timedelta
from faker import Faker

fake = Faker()
Faker.seed(42)

# Pre-generate a pool of garage IDs to reuse across officers (realistic —
# garages are shared across many officers in real life)
GARAGE_POOL = [f"GAR-{i:04d}" for i in range(1, 201)]  # 200 garages total

# Population stats from Kaggle (Stage 1, Step 7) — used to ground claim amounts
CLAIM_AMOUNT_MEAN = 52761.94
CLAIM_AMOUNT_STD = 26401.53


def generate_claims_for_officer_day(officer_id: str, sim_date: date, garage_concentration: float,
                                      preferred_garage: str, after_hours_pct: float,
                                      rng: np.random.Generator) -> list[dict]:
    """
    Generates individual claim records for one officer on one day.
    ...
    after_hours_pct: this officer's current D6 tendency (0-1) -- higher
    values make approval_hour more likely to fall outside 9AM-6PM,
    keeping the individual claims consistent with the daily D6 summary,
    same pattern used for D2/garage_concentration above.
    """
    n_claims_today = rng.poisson(0.18)

    claims = []
    for _ in range(n_claims_today):
        if rng.random() < garage_concentration:
            garage_id = preferred_garage
        else:
            garage_id = rng.choice(GARAGE_POOL)

        claim_amount = max(100.0, rng.normal(CLAIM_AMOUNT_MEAN, CLAIM_AMOUNT_STD))
        processing_days = max(1, rng.normal(7, 3))

        # NEW -- approval_hour, consistent with this officer's D6 tendency
        if rng.random() < after_hours_pct:
            approval_hour = int(rng.choice(list(range(0, 9)) + list(range(18, 24))))
        else:
            approval_hour = int(rng.integers(9, 18))

        claims.append({
            "claim_id": f"CLM-{fake.unique.random_number(digits=8, fix_len=True)}",
            "officer_id": officer_id,
            "claim_date": sim_date,
            "claim_amount": round(claim_amount, 2),
            "garage_id": garage_id,
            "processing_days": round(processing_days, 1),
            "approval_hour": approval_hour,  # NEW
        })

    return claims

# ── Baseline values for a clean officer (day 1 starting point) ──
# Grounded loosely in realistic ranges; d1/d4 use different scales than d2/d3/d5/d6
BASELINE = {
    "d1_approval_rate": 0.78,
    "d2_garage_concentration": 0.15,
    "d3_threshold_proximity": 0.08,
    "d4_surveyor_diversity": 12.0,
    "d5_turnaround_variance": 3.0,
    "d6_after_hours_pct": 0.03,
}

# Noise level (std dev of random daily jitter) per dimension — small, since
# behavior shouldn't swing wildly day to day for anyone, clean or not.
NOISE_STD = {
    "d1_approval_rate": 0.01,
    "d2_garage_concentration": 0.01,
    "d3_threshold_proximity": 0.005,
    "d4_surveyor_diversity": 0.3,
    "d5_turnaround_variance": 0.15,
    "d6_after_hours_pct": 0.003,
}

# Daily drift amount applied AFTER collusion_start_day, per archetype.
# Positive = dimension rises, negative = dimension falls (e.g. D4 collapses).
DRIFT_PER_DAY = {
    "gradually_colluding": {
        "d1_approval_rate": 0.003,
        "d2_garage_concentration": 0.012,
        "d3_threshold_proximity": 0.006,
        "d4_surveyor_diversity": -0.15,
        "d5_turnaround_variance": 0.03,
        "d6_after_hours_pct": 0.002,
    },
    "actively_fraudulent": {
        "d1_approval_rate": 0.006,
        "d2_garage_concentration": 0.025,
        "d3_threshold_proximity": 0.014,
        "d4_surveyor_diversity": -0.35,
        "d5_turnaround_variance": 0.07,
        "d6_after_hours_pct": 0.005,
    },
}

DIMENSIONS = ["d1_approval_rate", "d2_garage_concentration", "d3_threshold_proximity",
              "d4_surveyor_diversity", "d5_turnaround_variance", "d6_after_hours_pct"]

# Realistic bounds so a random walk can't drift into nonsense (e.g. rate > 1.0)
BOUNDS = {
    "d1_approval_rate": (0.0, 1.0),
    "d2_garage_concentration": (0.0, 1.0),
    "d3_threshold_proximity": (0.0, 1.0),
    "d4_surveyor_diversity": (1.0, 30.0),
    "d5_turnaround_variance": (0.1, 20.0),
    "d6_after_hours_pct": (0.0, 1.0),
}


def simulate_officer(officer_id: str, archetype: str, rng: np.random.Generator) -> pd.DataFrame:
    """
    Simulates one officer's 60-day behavioral history as a day-by-day random walk.

    archetype: "clean", "gradually_colluding", or "actively_fraudulent"

    Returns a DataFrame with 60 rows (one per day) and columns:
    officer_id, day, date, d1..d6 (raw values), archetype, collusion_start_day
    """
    # Decide when drift begins (irrelevant for "clean" officers)
    if archetype == "clean":
        collusion_start_day = None
    elif archetype == "gradually_colluding":
        collusion_start_day = int(rng.integers(15, 40))  # starts sometime days 15-40
    else:  # actively_fraudulent
        collusion_start_day = int(rng.integers(5, 20))   # starts earlier, more aggressive

    # Start at baseline, with a small per-officer random offset (so officers
    # aren't all identical clones on day 1)
    current = {
        dim: BASELINE[dim] + rng.normal(0, NOISE_STD[dim] * 2)
        for dim in DIMENSIONS
    }

    rows = []
    start_date = date(2026, 6, 1)  # arbitrary fixed start date for the simulation

    for day in range(1, 61):
        # Apply drift ONLY if this officer is past their collusion_start_day
        if archetype != "clean" and collusion_start_day is not None and day >= collusion_start_day:
            drift = DRIFT_PER_DAY[archetype]
            for dim in DIMENSIONS:
                current[dim] += drift[dim]

        # Apply random daily noise to EVERY officer, every day (this is what
        # makes it a random walk, not a straight line)
        row = {}
        for dim in DIMENSIONS:
            noisy_value = current[dim] + rng.normal(0, NOISE_STD[dim])
            lo, hi = BOUNDS[dim]
            noisy_value = max(lo, min(hi, noisy_value))  # clip to realistic bounds
            row[dim] = noisy_value
            current[dim] = noisy_value  # carry the noisy value forward (this IS the walk)

        row["officer_id"] = officer_id
        row["day"] = day
        row["date"] = start_date + timedelta(days=day - 1)
        row["archetype"] = archetype
        row["collusion_start_day"] = collusion_start_day
        rows.append(row)

    return pd.DataFrame(rows)

def generate_full_population(n_officers: int = 10000, seed: int = 42):
    """
    Generates the full synthetic officer population AND their underlying
    claim-level records (Fix A-2 — synthetic_claims table).

    Returns: (officer_twin_df, claims_df)
    """
    rng = np.random.default_rng(seed=seed)
    fake_local = Faker()
    Faker.seed(seed)

    n_clean = int(n_officers * 0.70)
    n_colluding = int(n_officers * 0.20)
    n_fraudulent = n_officers - n_clean - n_colluding

    archetypes = (
        ["clean"] * n_clean +
        ["gradually_colluding"] * n_colluding +
        ["actively_fraudulent"] * n_fraudulent
    )
    rng.shuffle(archetypes)

    all_officer_dfs = []
    all_claims = []

    for i, archetype in enumerate(archetypes):
        officer_id = f"OFF-{i+1:06d}"
        officer_df = simulate_officer(officer_id, archetype, rng)
        all_officer_dfs.append(officer_df)

        # Each officer gets ONE preferred garage (used more as their d2 rises)
        preferred_garage = rng.choice(GARAGE_POOL)

        for _, row in officer_df.iterrows():
            claims = generate_claims_for_officer_day(
                officer_id=officer_id,
                sim_date=row["date"],
                garage_concentration=row["d2_garage_concentration"],
                preferred_garage=preferred_garage,
                after_hours_pct=row["d6_after_hours_pct"],  # NEW
                rng=rng,
            )
            all_claims.extend(claims)

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i+1}/{n_officers} officers...")

    officer_twin_df = pd.concat(all_officer_dfs, ignore_index=True)
    claims_df = pd.DataFrame(all_claims)
    return officer_twin_df, claims_df


if __name__ == "__main__":
    print("Generating synthetic officer population (10,000 officers x 60 days)...")
    df, claims_df = generate_full_population(n_officers=10000, seed=42)

    print(f"\nTotal officer-day rows: {len(df)}")
    print(f"Unique officers: {df['officer_id'].nunique()}")
    print(f"\nArchetype breakdown:")
    print(df.groupby("archetype")["officer_id"].nunique())

    print(f"\nTotal claims generated: {len(claims_df)}")
    print(f"Sample claims:")
    print(claims_df.head())

    df.to_csv("data_utils/raw_data/synthetic_officers.csv", index=False)
    claims_df.to_csv("data_utils/raw_data/synthetic_claims.csv", index=False)
    print(f"\nSaved synthetic_officers.csv and synthetic_claims.csv")





    

