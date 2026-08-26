import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from agents.agent_b_anomaly.models.autoencoder import SymmetricAutoencoder


def train_and_calibrate(
    data_path: str = "data_utils/raw_data/synthetic_officers.csv",
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Place Person A's synthetic_officers.csv in data_utils/raw_data/"
        )

    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # Strict baseline filtering: Raise error instead of falling back to contaminated rows
    if "archetype" in df.columns:
        clean_df = df[df["archetype"].isin([0, "0", "clean", "Clean", "CLEAN"])].copy()
    elif "fraud_label" in df.columns:
        clean_df = df[df["fraud_label"].isin([0, "0", False, "false", "False"])].copy()
    else:
        raise KeyError("Dataset lacks recognizable 'archetype' or 'fraud_label' column.")

    if len(clean_df) == 0:
        raise ValueError("Strict Filter Failure: Zero clean baseline records identified. Refusing to train on unverified data.")

    print(f"Identified {len(clean_df)} verified clean baseline profile rows.")

    raw_cols = [
        "d1_approval_rate",
        "d2_garage_concentration",
        "d3_threshold_proximity",
        "d4_surveyor_diversity",
        "d5_turnaround_variance",
        "d6_after_hours_pct",
    ]

    z_cols = [f"{c}_z" for c in raw_cols]
    has_z_cols = all(c in clean_df.columns for c in z_cols)

    if has_z_cols:
        X = clean_df[z_cols].values.astype(np.float32)
    else:
        norm_arrays = []
        for raw in raw_cols:
            vals = clean_df[raw].values.astype(np.float32)
            mean = np.mean(vals)
            std = np.std(vals)
            std = std if std > 1e-6 else 1e-6
            norm_arrays.append(((vals - mean) / std).reshape(-1, 1))
        X = np.hstack(norm_arrays)

    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)

    actual_batch_size = min(batch_size, len(train_tensor))
    train_dataset = torch.utils.data.TensorDataset(train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=actual_batch_size, shuffle=True
    )

    model = SymmetricAutoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - MSE Loss: {total_loss / len(X_train):.6f}")

    model.eval()
    with torch.no_grad():
        val_recon = model(val_tensor)
        val_errors = torch.mean((val_tensor - val_recon) ** 2, dim=1).numpy()

    p75_threshold = float(np.percentile(val_errors, 75))

    os.makedirs("agents/agent_b_anomaly/weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    torch.save(model.state_dict(), "agents/agent_b_anomaly/weights/autoencoder.pt")
    np.save("agents/agent_b_anomaly/weights/baseline_errors.npy", val_errors)

    with open("results/ae_calibration.md", "w") as f:
        f.write(f"# Autoencoder Calibration Report\n\n")
        f.write(f"- **Clean Training Instances:** {len(X_train)}\n")
        f.write(f"- **Validation Instances:** {len(X_val)}\n")
        f.write(f"- **Calibrated 75th Percentile MSE Cutoff:** {p75_threshold:.6f}\n")
        f.write(f"- **Mean Validation MSE:** {float(np.mean(val_errors)):.6f}\n")
        f.write(f"- **Max Validation MSE:** {float(np.max(val_errors)):.6f}\n")

    print(f"\nCalibration artifact saved to results/ae_calibration.md")


if __name__ == "__main__":
    train_and_calibrate()