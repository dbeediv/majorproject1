import pandas as pd
import torch
import joblib
import sys
sys.path.insert(0, "models/agent1")
from lstm_model import LSTMAutoencoder, build_sequence_tensors, normalize_sequences

df = pd.read_csv("data_utils/raw_data/synthetic_officers.csv")
sequences, officer_ids, archetypes = build_sequence_tensors(df)
norm_data = joblib.load("models/agent1/lstm_normalization.pkl")
normalized_sequences = (sequences - norm_data["means"]) / norm_data["stds"]

model = LSTMAutoencoder()
model.load_state_dict(torch.load("models/agent1/lstm_model.pt"))
model.eval()

# Pick one fraudulent officer and compute DAY-BY-DAY error (not just overall average)
fraud_indices = [i for i, a in enumerate(archetypes) if a == "actively_fraudulent"]
sample_idx = fraud_indices[0]
sample_id = officer_ids[sample_idx]

officer_seq = normalized_sequences[sample_idx].unsqueeze(0)
with torch.no_grad():
    reconstructed = model(officer_seq)
    daily_error = ((reconstructed - officer_seq) ** 2).mean(dim=2).squeeze().numpy()

collusion_day = df[df["officer_id"] == sample_id]["collusion_start_day"].iloc[0]

print(f"Officer {sample_id} (actively_fraudulent), collusion starts day {collusion_day}")
print(f"\nDaily reconstruction error, before vs after collusion start:")
print(f"  Days 1 to {collusion_day-1} (before): avg error = {daily_error[:int(collusion_day)-1].mean():.4f}")
print(f"  Days {collusion_day} to 60 (after):  avg error = {daily_error[int(collusion_day)-1:].mean():.4f}")