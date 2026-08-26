import pandas as pd


def load_and_clean_kaggle_data(filepath: str = "data_utils/raw_data/insurance_claims.xlsx") -> pd.DataFrame:
    """
    Loads the raw Kaggle Auto Insurance Fraud dataset and cleans it.
    Returns a cleaned DataFrame ready for population-stat extraction (Step 7)
    and officer assignment (Step 8).
    """
    df = pd.read_excel(filepath)

    df["authorities_contacted"] = df["authorities_contacted"].fillna("None")
    df["fraud_reported"] = df["fraud_reported"].map({"Y": 1, "N": 0})
    df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"])
    df["incident_date"] = pd.to_datetime(df["incident_date"])

    return df


def extract_population_stats(df: pd.DataFrame) -> dict:
    """
    Extracts summary statistics from the cleaned Kaggle data.
    These seed the synthetic officer generator (Stage 2) and the
    D1-D6 dimension functions (Stage 3) so synthetic claim amounts,
    timing patterns, and thresholds are grounded in real data.
    """
    stats = {
        "claim_amount_mean": df["total_claim_amount"].mean(),
        "claim_amount_std": df["total_claim_amount"].std(),
        "claim_amount_min": df["total_claim_amount"].min(),
        "claim_amount_max": df["total_claim_amount"].max(),
        "claim_threshold_75th": df["total_claim_amount"].quantile(0.75),  # NEW — D3 escalation threshold
        "fraud_rate": df["fraud_reported"].mean(),
        "incident_hour_mean": df["incident_hour_of_the_day"].mean(),
        "incident_hour_std": df["incident_hour_of_the_day"].std(),
    }
    return stats


if __name__ == "__main__":
    df = load_and_clean_kaggle_data()
    print("Cleaned shape:", df.shape)
    print("Missing values remaining:", df.isnull().sum().sum())
    print("Fraud rate:", df["fraud_reported"].mean())
    print(df[["authorities_contacted", "fraud_reported"]].head())

    print("\n=== Population stats (for synthetic generator) ===")
    stats = extract_population_stats(df)
    for k, v in stats.items():
        print(f"{k}: {v:.2f}")