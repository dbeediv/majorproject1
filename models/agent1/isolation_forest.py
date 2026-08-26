import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

DIMENSIONS = ["d1_approval_rate", "d2_garage_concentration", "d3_threshold_proximity",
              "d4_surveyor_diversity", "d5_turnaround_variance", "d6_after_hours_pct"]


def compute_population_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds population-level z-score columns for each D1-D6 dimension.
    Every officer-day is compared against the OVERALL population mean/std
    (not personal history -- that's Stage 6's job).
    """
    df = df.copy()
    for dim in DIMENSIONS:
        mean = df[dim].mean()
        std = df[dim].std()
        df[f"{dim}_z"] = (df[dim] - mean) / (std + 1e-9)  # +epsilon avoids divide-by-zero
    return df


def train_isolation_forest(df_z: pd.DataFrame, contamination: float = 0.1, random_state: int = 42):
    """
    Trains ONE global Isolation Forest on z-scored D1-D6 features across
    ALL officers (not per-officer -- see design note in reference doc).

    contamination: expected proportion of anomalies in the data. We set
    this to 0.1 as a starting point (our archetypes are 30% non-clean,
    but the DAILY behavior of colluding officers looks normal for a while
    before drift kicks in, so the true anomalous-day rate is lower than 30%).
    """
    z_cols = [f"{dim}_z" for dim in DIMENSIONS]
    X = df_z[z_cols].values

    model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    model.fit(X)

    # decision_function: higher = more normal, lower (more negative) = more anomalous
    df_z = df_z.copy()
    df_z["if_anomaly_score"] = model.decision_function(X)
    df_z["if_predicted_anomaly"] = model.predict(X)  # -1 = anomaly, 1 = normal

    return model, df_z


if __name__ == "__main__":
    print("Loading synthetic officer data...")
    df = pd.read_csv("data_utils/raw_data/synthetic_officers.csv")
    print(f"Loaded {len(df)} rows")

    print("\nComputing population z-scores...")
    df_z = compute_population_zscores(df)

    z_cols = [f"{dim}_z" for dim in DIMENSIONS]
    print("\nZ-score summary (should be ~mean 0, std 1 for each):")
    print(df_z[z_cols].describe().loc[["mean", "std"]])

    print("\nTraining Isolation Forest...")
    model, df_scored = train_isolation_forest(df_z)
    print("Training complete.")

    print("\nAnomaly score by archetype (lower avg = more anomalous):")
    print(df_scored.groupby("archetype")["if_anomaly_score"].mean())

    print("\nPredicted anomaly rate by archetype (-1 = flagged as anomaly):")
    print(df_scored.groupby("archetype")["if_predicted_anomaly"].apply(lambda x: (x == -1).mean()))

    print("\nSaving model and scored data...")
    joblib.dump(model, "models/agent1/isolation_forest_model.pkl")
    df_scored.to_csv("data_utils/raw_data/synthetic_officers_scored.csv", index=False)
    print("Saved isolation_forest_model.pkl and synthetic_officers_scored.csv")