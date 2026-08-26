import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DIMENSIONS = ["d1_approval_rate", "d2_garage_concentration", "d3_threshold_proximity",
              "d4_surveyor_diversity", "d5_turnaround_variance", "d6_after_hours_pct"]


def build_sequence_tensors(df: pd.DataFrame):
    """
    Converts the flat officer-day DataFrame into a 3D tensor:
    shape (n_officers, 60, 6) -- one full 60-day sequence per officer.

    Returns: (tensor, officer_ids, archetypes) so we can trace results
    back to specific officers later for validation.
    """
    officer_ids = sorted(df["officer_id"].unique())
    n_officers = len(officer_ids)
    n_days = 60
    n_dims = len(DIMENSIONS)

    sequences = np.zeros((n_officers, n_days, n_dims), dtype=np.float32)
    archetypes = []

    df_indexed = df.set_index(["officer_id", "day"])

    for i, officer_id in enumerate(officer_ids):
        officer_data = df[df["officer_id"] == officer_id].sort_values("day")
        sequences[i] = officer_data[DIMENSIONS].values
        archetypes.append(officer_data["archetype"].iloc[0])

    return torch.tensor(sequences), officer_ids, archetypes


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_dims=6, hidden_size=16, latent_size=8):
        super().__init__()
        # Encoder: reads the 60-day sequence, compresses to a small hidden state
        self.encoder_lstm = nn.LSTM(input_size=n_dims, hidden_size=hidden_size, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_size, latent_size)

        # Decoder: takes the compressed representation, tries to rebuild 60 days
        self.decoder_fc = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(input_size=hidden_size, hidden_size=n_dims, batch_first=True)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Encode: compress the sequence into a single latent vector
        _, (h_n, _) = self.encoder_lstm(x)
        latent = self.encoder_fc(h_n.squeeze(0))  # shape: (batch, latent_size)

        # Decode: expand the latent vector back out to 60 timesteps
        decoder_input = self.decoder_fc(latent).unsqueeze(1).repeat(1, seq_len, 1)
        reconstructed, _ = self.decoder_lstm(decoder_input)

        return reconstructed


def train_lstm_autoencoder(sequences: torch.Tensor, archetypes: list, epochs: int = 15, batch_size: int = 64):
    """
    Trains the LSTM autoencoder on CLEAN officers only (per design note above).
    Compressed for today's timeline: fewer epochs than a full production run,
    but enough to demonstrate the reconstruction-error signal works.
    """
    clean_mask = [a == "clean" for a in archetypes]
    clean_sequences = sequences[clean_mask]
    print(f"Training on {len(clean_sequences)} clean officer sequences...")

    model = LSTMAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    n_samples = len(clean_sequences)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples)
        total_loss = 0.0

        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i:i + batch_size]
            batch = clean_sequences[batch_idx]

            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)

        avg_loss = total_loss / n_samples
        print(f"  Epoch {epoch+1}/{epochs} — avg reconstruction loss: {avg_loss:.6f}")

    return model

def normalize_sequences(sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Z-scores each of the 6 dimensions across the whole dataset, so no
    single dimension's raw scale dominates the LSTM's reconstruction loss.

    Returns: (normalized_sequences, means, stds) -- means/stds saved so we
    can un-normalize later if needed, and reused consistently at inference.
    """
    # Compute mean/std per dimension across ALL officers and ALL days
    means = sequences.mean(dim=(0, 1), keepdim=True)  # shape (1, 1, 6)
    stds = sequences.std(dim=(0, 1), keepdim=True) + 1e-9
    normalized = (sequences - means) / stds
    return normalized, means, stds


def compute_reconstruction_errors(model: LSTMAutoencoder, normalized_sequences: torch.Tensor) -> torch.Tensor:
    """
    Runs every officer's sequence through the trained model and computes
    per-officer reconstruction error (mean squared error across all 60
    days and 6 dimensions). Higher error = more unusual/anomalous sequence.
    """
    model.eval()
    with torch.no_grad():
        reconstructed = model(normalized_sequences)
        # Mean squared error per officer, across all days and dimensions
        errors = ((reconstructed - normalized_sequences) ** 2).mean(dim=(1, 2))
    return errors


if __name__ == "__main__":
    print("Loading synthetic officer data...")
    df = pd.read_csv("data_utils/raw_data/synthetic_officers.csv")

    print("Building sequence tensors...")
    sequences, officer_ids, archetypes = build_sequence_tensors(df)

    print("Normalizing sequences...")
    normalized_sequences, seq_means, seq_stds = normalize_sequences(sequences)

    print("\nTraining LSTM autoencoder on clean officers (normalized data)...")
    model = train_lstm_autoencoder(normalized_sequences, archetypes, epochs=15)
    print("Training complete.")

    print("\nComputing reconstruction errors for ALL officers...")
    errors = compute_reconstruction_errors(model, normalized_sequences)

    results_df = pd.DataFrame({
        "officer_id": officer_ids,
        "archetype": archetypes,
        "lstm_reconstruction_error": errors.numpy(),
    })

    print("\nAverage reconstruction error by archetype (higher = more anomalous):")
    print(results_df.groupby("archetype")["lstm_reconstruction_error"].mean())

    # Save the model and results for Stage 6
    import joblib
    torch.save(model.state_dict(), "models/agent1/lstm_model.pt")
    joblib.dump({"means": seq_means, "stds": seq_stds}, "models/agent1/lstm_normalization.pkl")
    results_df.to_csv("data_utils/raw_data/lstm_reconstruction_errors.csv", index=False)
    print("\nSaved lstm_model.pt, lstm_normalization.pkl, and lstm_reconstruction_errors.csv")