from typing import List, Union
import numpy as np
import torch
import torch.nn as nn


class SymmetricAutoencoder(nn.Module):
    """
    Symmetric undercomplete bottleneck Autoencoder (6 -> 4 -> 2 -> 4 -> 6)[cite: 4].
    Trained strictly on normal/clean officer behavioral feature vectors[cite: 2, 4].
    """
    def __init__(self):
        super().__init__()
        # Encoder: Compresses 6 normalized dimensions to a 2D latent space[cite: 4]
        self.encoder = nn.Sequential(
            nn.Linear(6, 4),
            nn.LeakyReLU(0.1),
            nn.Linear(4, 2),
            nn.LeakyReLU(0.1),
        )
        # Decoder: Symmetrically reconstructs the 6D feature vector from latent space[cite: 4]
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.LeakyReLU(0.1),
            nn.Linear(4, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class AutoencoderScorer:
    """
    Evaluates reconstruction error of input feature vectors and maps raw MSE
    to an empirical percentile anomaly rank [0.0 - 1.0] using baseline validation errors[cite: 4].
    """
    def __init__(
        self,
        model: SymmetricAutoencoder,
        baseline_errors: Union[np.ndarray, List[float]],
    ):
        self.model = model.eval()
        # Sort baseline validation errors for fast rank lookups via binary search
        if isinstance(baseline_errors, list):
            baseline_errors = np.array(baseline_errors, dtype=np.float32)
        
        if len(baseline_errors) > 0:
            self.baseline_percentiles = np.sort(baseline_errors)
        else:
            self.baseline_percentiles = np.array([], dtype=np.float32)

    def compute_raw_mse(self, z_vector: List[float]) -> float:
        """Computes raw Mean Squared Error reconstruction loss[cite: 1, 2]."""
        tensor_in = torch.tensor(z_vector, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            recon = self.model(tensor_in)
            mse = nn.functional.mse_loss(recon, tensor_in).item()
        return float(mse)

    def score(self, z_vector: List[float]) -> float:
        """
        Calculates the continuous reconstruction anomaly percentile score [0.0 to 1.0][cite: 4].
        0.0 = completely normal baseline behavior; 1.0 = extreme anomaly[cite: 4].
        """
        mse = self.compute_raw_mse(z_vector)
        
        # Fallback if no baseline array is provided
        if len(self.baseline_percentiles) == 0:
            return float(np.clip(mse, 0.0, 1.0))

        # Determine percentile rank of current MSE against clean baseline distribution[cite: 4]
        rank = np.searchsorted(self.baseline_percentiles, mse)
        percentile_score = rank / len(self.baseline_percentiles)
        
        return float(np.clip(percentile_score, 0.0, 1.0))