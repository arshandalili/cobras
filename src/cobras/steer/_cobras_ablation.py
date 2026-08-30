"""Ablation variants of COBRAS.

Every switch here exists to turn *off* or *replace* one part of the shipped method so that
its contribution can be measured. None of it is on the paper's path: `AblationCOBRAS()` with
no arguments is numerically identical to `COBRAS()`, and each knob's default is the shipped
behaviour. The point of the separate class is that `COBRAS` in `_cobras.py` stays readable
as the method, and a config that reaches for an ablation has to say so by name.

Grouped by what they ablate:

  potentials       "plain" keeps the Sinkhorn potentials attached to their own samples
                   (Eqs. 10-11) instead of re-marginalizing them at the query (Eqs. 15-16)
  drift            "gradient" keeps the 1/sigma^2 factor of Eq. (31) in the drift instead of
                   absorbing it into the step size
  step_mode        "raw" lets the drift magnitude set the distance travelled, rather than
                   normalizing to a unit-length Euler step
  bandwidth        "fixed" uses the fitted sigma^2 everywhere instead of the query-adaptive
                   bandwidth of Eq. (18)
  bandwidth_scale  multiplies whichever bandwidth is in force, sharpening or flattening the
                   selection the Eq. (18) weights make among the contrastive samples
  uniform_weights  replaces those weights with 1/N, keeping the geometry but dropping the
                   selection, so the step is a plain difference of Riemannian centroids
  abstain_signal   what the gate measures: k-NN radius (shipped), the bridge's own marginal,
                   the kernel density, or the inverse drift norm
  abstain_calibration  "quantile" maps the score through the fitted empirical CDF instead of
                   the shipped soft ratio threshold
  abstain_bandwidth_scale  multiplies sigma^2 inside `log_marginal` only
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from ._cobras import COBRAS, _EPS, _SAME_POINT_THR
from ..utils.sphere import knn_geodesic_dist


class AblationCOBRAS(COBRAS):
    def __init__(
        self,
        k_bw: int = 5,
        n_sinkhorn: int = 5,
        alpha_sigma: float = 1e-3,
        epsilon: float = 0.0,
        max_iters: int = 10,
        vmf_kappa: float | None = None,
        vmf_beta: float = 0.0,
        abstain_percentile: float | None = None,
        abstain_k: int = 32,
        abstain_sharpness: float = 50.0,
        abstain_on_queries: bool = False,
        abstain_signal: Literal["knn", "marginal", "density", "drift"] = "knn",
        abstain_calibration: Literal["ratio", "quantile"] = "ratio",
        abstain_bandwidth_scale: float = 1.0,
        step_mode: Literal["unit", "raw"] = "unit",
        bandwidth: Literal["adaptive", "fixed"] = "adaptive",
        bandwidth_scale: float = 1.0,
        uniform_weights: bool = False,
        potentials: Literal["extended", "plain"] = "extended",
        drift: Literal["centroid", "gradient"] = "centroid",
    ) -> None:
        super().__init__(
            k_bw=k_bw, n_sinkhorn=n_sinkhorn, alpha_sigma=alpha_sigma, epsilon=epsilon,
            max_iters=max_iters, vmf_kappa=vmf_kappa, vmf_beta=vmf_beta,
            abstain_percentile=abstain_percentile, abstain_k=abstain_k,
            abstain_sharpness=abstain_sharpness, abstain_on_queries=abstain_on_queries,
        )
        self.abstain_signal = abstain_signal
        self.abstain_calibration = abstain_calibration
        self.abstain_bandwidth_scale = float(abstain_bandwidth_scale)
        self.step_mode = step_mode
        self.bandwidth = bandwidth
        self.bandwidth_scale = float(bandwidth_scale)
        self.uniform_weights = bool(uniform_weights)
        self.potentials = potentials
        self.drift = drift

        self.step_scale: float | None = None

    # ---------------------------------------------------------------- fitting

    def _abstain_reference(self, H_all: Tensor, ref_X: Tensor | None) -> Tensor:
        if self.abstain_on_queries:
            return super()._abstain_reference(H_all, ref_X)
        if self.abstain_signal == "knn":
            return knn_geodesic_dist(
                self.h_neg, self.R, k=min(self.abstain_k, self.h_neg.size(0) - 1)
            )
        # a non-k-NN signal has no leave-one-out form, so it is scored on the fitted set
        return self._abstain_score(H_all)

    def _post_fit(self, H_all: Tensor) -> None:
        if self.step_mode == "raw":
            self.step_scale = self._calibrate_step_scale(H_all)

    @torch.no_grad()
    def _calibrate_step_scale(self, H_all: Tensor, n_ref: int = 256, chunk: int = 32) -> float:
        """Arc length per unit drift norm, so that a median in-distribution query
        travels the same distance as it would under the unit-step schedule."""
        idx = torch.randperm(H_all.size(0), generator=torch.Generator().manual_seed(0))[:n_ref]
        H_ref = H_all[idx.to(H_all.device)]
        V = torch.cat(
            [self._field(H_ref[i : i + chunk])[0] for i in range(0, H_ref.size(0), chunk)], 0
        )
        cos_T0 = ((H_ref / self.R) * self.mu_T).sum(-1).clamp(-1 + _EPS, 1 - _EPS)
        theta_ref = torch.acos(cos_T0).median()
        return float((theta_ref * self.R / V.norm(dim=-1).median().clamp(min=_EPS)).item())

    # ------------------------------------------------------------ the bridge

    def _bandwidth(self, dist2: Tensor) -> Tensor | float:
        if self.bandwidth == "fixed":
            return self.bandwidth_scale * self.sigma2
        return self.bandwidth_scale * super()._bandwidth(dist2)

    def _potentials(self, q: Tensor) -> tuple[Tensor, Tensor]:
        if self.potentials == "plain":
            # Eqs. (10)-(11) as the theory states them: the Sinkhorn potentials stay attached
            # to their own samples and the kernel does the extending, so psi_hat is already
            # defined at any q and Eq. (31) is the exact Riemannian gradient of log psi_hat.
            return self.log_psi, self.log_phi
        return super()._potentials(q)

    def _kernel_weights(self, q: Tensor, H: Tensor, log_w: Tensor):
        wK, theta, cos_t, log_Z, sigma2 = super()._kernel_weights(q, H, log_w)
        if self.uniform_weights:
            wK = torch.full_like(wK, 1.0 / wK.shape[-1])
        return wK, theta, cos_t, log_Z, sigma2

    def _combine_drift(
        self, V_pos: Tensor, V_neg: Tensor, s2_pos: Tensor | float, s2_neg: Tensor | float
    ) -> Tensor:
        if self.drift == "gradient":
            # keep the 1/sigma^2 factor of Eq. (31) instead of absorbing it into the step
            # size: with a query-adaptive bandwidth it is not a constant
            return V_pos / s2_pos - V_neg / s2_neg
        return super()._combine_drift(V_pos, V_neg, s2_pos, s2_neg)

    # -------------------------------------------------------- strength & gate

    @torch.no_grad()
    def _abstain_score(self, q: Tensor, chunk: int = 64) -> Tensor:
        """Distance-like score for how far a query sits from the transport support.

        The drift of Eq. (13) is a ratio of the two potentials, so their kernel decay in the
        distance to the data cancels. It survives only in the product, which is the bridge's
        own marginal p_0 = psi_hat * phi_hat (Sec. 4.2).
        """
        if self.abstain_signal == "knn":
            return super()._abstain_score(q, chunk=chunk)

        if self.abstain_signal == "marginal":
            return -self.log_marginal(
                q, raw=True, bandwidth_scale=self.abstain_bandwidth_scale
            )

        bandwidth, drift = self.bandwidth, self.drift
        if self.abstain_signal == "density":
            self.bandwidth = "fixed"
        self.drift = "gradient"
        try:
            out = []
            for i in range(0, q.size(0), chunk):
                qi = q[i : i + chunk]
                if self.abstain_signal == "density":
                    _, log_psi_hat, _ = self._weighted_centroid(
                        qi, self.h_pos, self._query_log_psi(qi)
                    )
                    _, log_phi_hat, _ = self._weighted_centroid(
                        qi, self.h_neg, self._query_log_phi(qi)
                    )
                    out.append(-(log_psi_hat + log_phi_hat))
                else:
                    out.append(1.0 / self._field(qi)[0].norm(dim=-1).clamp(min=_EPS))
        finally:
            self.bandwidth, self.drift = bandwidth, drift
        return torch.cat(out)

    def _abstain_gate(self, q: Tensor) -> Tensor:
        if self.abstain_calibration != "quantile":
            return super()._abstain_gate(q)
        score = self._abstain_score(q)
        ref = self.abstain_ref.to(device=score.device, dtype=score.dtype)
        rank = torch.searchsorted(ref, score.contiguous()) / ref.numel()
        return ((1.0 - rank) / (1.0 - self.abstain_percentile)).clamp(0.0, 1.0)

    # ------------------------------------------------------------- steering

    def _step(self, V: Tensor, dt_vec: Tensor, T: float, strength: Tensor) -> Tensor:
        if self.step_mode != "raw":
            return super()._step(V, dt_vec, T, strength)
        # keep the drift magnitude: the bridge itself decides how far to move
        return (T * strength / self.max_iters).unsqueeze(-1) * self.step_scale * V

    # ------------------------------------------------------------- analysis

    @torch.no_grad()
    def query_stats(self, X: Tensor) -> dict[str, Tensor]:
        """Per-query quantities the bridge exposes at the steering site, for OOD analysis.

        Sweeps both bandwidth choices and both drift forms, which is why it lives here
        rather than on the shipped class.
        """
        self._to(X.device, X.dtype)
        R = self.R
        p0 = X * (R / X.norm(dim=-1, keepdim=True).clamp(min=_SAME_POINT_THR))

        stats = {}
        bandwidth, drift = self.bandwidth, self.drift
        try:
            for bw in ("adaptive", "fixed"):
                self.bandwidth = bw
                V_pos, log_psi_hat, s2_pos = self._weighted_centroid(
                    p0, self.h_pos, self._query_log_psi(p0)
                )
                V_neg, log_phi_hat, s2_neg = self._weighted_centroid(
                    p0, self.h_neg, self._query_log_phi(p0)
                )
                # the potentials pay for distance; the product keeps it, the ratio cancels it
                stats[f"log_product_{bw}"] = log_psi_hat + log_phi_hat
                stats[f"log_ratio_{bw}"] = log_psi_hat - log_phi_hat
                stats[f"grad_psi_{bw}"] = (V_pos / s2_pos).norm(dim=-1)
                stats[f"grad_phi_{bw}"] = (V_neg / s2_neg).norm(dim=-1)
                for dr in ("centroid", "gradient"):
                    self.drift = dr
                    stats[f"drift_{dr}_{bw}"] = self._field(p0)[0].norm(dim=-1)
        finally:
            self.bandwidth, self.drift = bandwidth, drift

        cos_qn = ((p0 @ self.h_neg.T) / (R ** 2)).clamp(-1.0 + _EPS, 1.0 - _EPS)
        dist_qn = R * torch.acos(cos_qn)
        k = min(self.abstain_k, self.h_neg.size(0))
        stats["knn_radius"] = dist_qn.topk(k, dim=1, largest=False).values[:, -1]
        stats["nn_dist"] = dist_qn.min(dim=1).values
        stats["vmf_strength"] = self._compute_strength(p0)[0]
        return stats
