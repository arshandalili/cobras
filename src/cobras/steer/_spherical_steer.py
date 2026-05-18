'''
Paper: Spherical Steering: Geometry-Aware Activation Rotation for Language Models
Reference: 
- Paper: https://arxiv.org/pdf/2602.08169
- Code: https://github.com/chili-lab/Spherical-Steering/tree/main
'''

import torch
from torch import Tensor
import torch.nn.functional as F

from ._base_steer import Steer


class SphericalSteer(Steer):
    def __init__(
        self, 
        kappa: float = 20.0, 
        alpha: float = 0.15, 
        beta: float = 0.1
    ):
        super().__init__()
        # will be set in fit function
        self.mu_T = None
        self.mu_H = None 
        self.cos_sim = None

        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta
    
    def fit(self, pos_X: torch.Tensor, neg_X: torch.Tensor):
        # Compute centroids
        mean_true = pos_X.mean(dim=0)
        mean_false = neg_X.mean(dim=0)

        # Compute difference vector (truthful direction)
        diff_vec = mean_true - mean_false

        # Normalize to create unit prototypes
        mu_T = F.normalize(diff_vec, dim=0)
        mu_H = -mu_T  # Antipodal prototype

        # Verify they are antipodal (cosine similarity)
        cos_sim = torch.dot(mu_T, mu_H)

        self.mu_T = mu_T
        self.mu_H = mu_H
        self.cos_sim = cos_sim

        return self
    

    def _spherical_geometric_logic(self, x, mu_T, mu_H, kappa, alpha, beta):
        """
        Core spherical steering logic for a single hidden state vector.
        
        This function implements the geometric steering operation:
        1. Compute vMF probabilities for truthful (T) and hallucination (H) prototypes
        2. If hallucination probability exceeds threshold, apply steering
        3. Steering rotates the vector toward the truthful prototype on the sphere
        
        Args:
            x: Hidden state vector [B, D]
            mu_T: Truthful prototype (unit vector) [D]
            mu_H: Hallucination prototype (unit vector) [D]
            kappa: vMF concentration parameter (higher = sharper decisions)
            alpha: Maximum steering strength (0 to 1)
            beta: Threshold for triggering steering (p_H - p_T > beta)
        
        Returns:
            x_new: Steered hidden state vector [B, D]
            triggered: Boolean indicating if steering was applied [B, 1]
        """
        orig_dtype = x.dtype
        x = x.float()
        mu_T = mu_T.float()
        mu_H = mu_H.float()
        
        # Preserve original norm for rescaling
        orig_norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        x_hat = x / orig_norm

        # Compute vMF log-likelihoods (proportional to cosine similarity)
        cos_T = (x_hat * mu_T.unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)
        cos_H = (x_hat * mu_H.unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)

        # Softmax to get probabilities
        logits = torch.stack([kappa * cos_T, kappa * cos_H], dim=-1)
        probs = F.softmax(logits, dim=-1)
        p_T = probs[:, 0]
        p_H = probs[:, 1]

        delta = p_H - p_T
        trigger_mask = delta > beta

        x_new = x.clone()

        if trigger_mask.any():
            # Compute steering strength (linear interpolation above threshold)
            t = alpha * (delta - beta) / (1.0 - beta)
            t = torch.clamp(t, 0.0, 1.0)

            # Compute angle from truthful prototype
            theta = torch.acos(cos_T)  # [B]
            valid_mask = trigger_mask & (theta >= 1e-4)

            if valid_mask.any():
                # Compute new angle (rotate toward mu_T)
                theta_new = (1.0 - t) * theta

                # Spherical interpolation (SLERP-like)
                sin_theta = torch.sin(theta).clamp_min(1e-12)
                u = (x_hat - cos_T.unsqueeze(-1) * mu_T.unsqueeze(0)) / sin_theta.unsqueeze(-1)
                
                # Compute new angle (rotate toward mu_T)
                x_new_hat = (
                    torch.cos(theta_new).unsqueeze(-1) * mu_T.unsqueeze(0)
                    + torch.sin(theta_new).unsqueeze(-1) * u
                )
                candidate = x_new_hat * orig_norm

                x_new[valid_mask] = candidate[valid_mask]

        x_new = x_new.to(orig_dtype)

        return x_new, trigger_mask

    def steer(self, X: Tensor, T: float = 1.0):
        """
            Steering function does not use T, it is dynamically injected
        """
        mu_T = self.mu_T.to(X.device)
        mu_H = self.mu_H.to(X.device)
        
        X_steered, is_steered = self._spherical_geometric_logic(
            X, mu_T, mu_H, self.kappa, self.alpha, self.beta
        )
        return X_steered

    def vector_field(self, X: Tensor):
        return
