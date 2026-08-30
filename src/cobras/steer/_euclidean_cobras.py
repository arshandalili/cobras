"""A Euclidean control for COBRAS.

`EuclideanCOBRAS` is COBRAS with the spherical geometry removed and nothing else changed,
which is what makes it a control for the question "does the sphere earn its place?":

  COBRAS (_cobras.py)                       EuclideanCOBRAS
  --------------------------------------    ---------------------------------------
  h <- R h/||h||   (radial projection)      h kept as is
  c_ji = d_S(h_j-, h_i+)^2 / (2 sigma^2)    c_ji = ||h_j- - h_i+||^2 / (2 sigma^2)
  sigma^2 from kNN geodesic distance        sigma^2 from kNN Euclidean distance
  KDE weights use d_S(q, h)^2               KDE weights use ||q - h||^2
  drift = sum_i w_i log_q(h_i)              drift = sum_i w_i (h_i - q)
  tangential projection of the drift        no tangential projection
  q <- exp_q(dt * V/||V||)                  q <- q + dt * V/||V||
  final rescale to ||h_q||                  no rescale

Identical in both: the Sinkhorn solve, the extended (query-marginalised) potentials, the
adaptive bandwidth rule, the vMF strength gate, the number of Euler steps K, and the step
length schedule dt = T * s * theta_0 * R / K, so the two updates travel the same distance.

It subclasses `AblationCOBRAS` because it reads the same `potentials` / `drift` /
`uniform_weights` switches; with their defaults it is the Euclidean twin of shipped COBRAS.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._cobras import _EPS, _SAME_POINT_THR
from ._cobras_ablation import AblationCOBRAS


def _sq_dists(A: Tensor, B: Tensor) -> Tensor:
    """Squared Euclidean distances, [n, m]."""
    return torch.cdist(A, B).pow(2)


class EuclideanCOBRAS(AblationCOBRAS):
    @torch.no_grad()
    def fit(
        self, pos_X: Tensor, neg_X: Tensor, ref_X: Tensor | None = None
    ) -> "EuclideanCOBRAS":
        pos = pos_X.detach().to(torch.float32)
        neg = neg_X.detach().to(torch.float32)
        R = torch.cat([pos, neg], 0).norm(dim=-1).mean().item()
        self.R = float(R)
        # no radial projection: the samples stay where the model put them
        self.h_pos, self.h_neg = pos, neg

        H_all = torch.cat([self.h_pos, self.h_neg], 0)
        k = min(self.k_bw, H_all.size(0) - 1)
        dmat = torch.cdist(H_all, H_all)
        dmat.fill_diagonal_(float("inf"))
        sigma2 = dmat.topk(k, dim=1, largest=False).values[:, -1].median().item() ** 2
        self.sigma2 = float(max(sigma2, (R * 1e-3) ** 2))

        self.cost = _sq_dists(self.h_neg, self.h_pos) / (2.0 * self.sigma2)
        self.log_psi, self.log_phi = self._sinkhorn(self.cost, self.n_sinkhorn)

        diff = self.h_pos.mean(0) - self.h_neg.mean(0)
        self.mu_T = diff / diff.norm().clamp(min=_EPS)
        self._device = pos.device

        if self.abstain_percentile is not None:
            dn = torch.cdist(self.h_neg, self.h_neg)
            dn.fill_diagonal_(float("inf"))
            ref = dn.topk(
                min(self.abstain_k, self.h_neg.size(0) - 1), dim=1, largest=False
            ).values[:, -1]
            self.rho_ref = float(torch.quantile(ref.float(), self.abstain_percentile).item())
            self.abstain_ref = ref.sort().values
        return self

    def _query_log_psi(self, q: Tensor) -> Tensor:
        d2 = _sq_dists(q, self.h_neg)
        log_alpha = self.log_phi.unsqueeze(0) - d2 / (2.0 * self._bandwidth(d2))
        return torch.logsumexp(log_alpha.unsqueeze(2) - self.cost.unsqueeze(0), dim=1)

    def _query_log_phi(self, q: Tensor) -> Tensor:
        d2 = _sq_dists(q, self.h_pos)
        log_beta = self.log_psi.unsqueeze(0) - d2 / (2.0 * self._bandwidth(d2))
        return torch.logsumexp(log_beta.unsqueeze(1) - self.cost.unsqueeze(0), dim=2)

    def _kernel_weights(self, q: Tensor, H: Tensor, log_w: Tensor):
        """Returns (wK, dist2, log_Z, sigma2). The tuple is one element shorter than the
        spherical one because there is no angle to carry; `_weighted_centroid` below is
        overridden to match, and callers outside this class only ever read `[0]`."""
        dist2 = _sq_dists(q, H)
        sigma2 = self._bandwidth(dist2)
        lw = log_w if log_w.dim() == 2 else log_w.unsqueeze(0)
        log_wK = lw - dist2 / (2.0 * sigma2)
        log_Z = torch.logsumexp(log_wK, dim=-1)
        wK = (log_wK - log_Z.unsqueeze(-1)).exp()
        if self.uniform_weights:
            wK = torch.full_like(wK, 1.0 / wK.shape[-1])
        return wK, dist2, log_Z, sigma2

    def _weighted_centroid(self, q: Tensor, H: Tensor, log_w: Tensor):
        wK, _, log_Z, sigma2 = self._kernel_weights(q, H, log_w)
        # Euclidean analogue of the weighted Riemannian mean direction: log_q(h) -> h - q
        numer = wK @ H - wK.sum(-1, keepdim=True) * q
        return numer / (1.0 + self.alpha_sigma), log_Z, sigma2

    def _field(self, q: Tensor):
        log_psi_q, log_phi_q = self._potentials(q)
        V_pos, log_psi_hat, s2_pos = self._weighted_centroid(q, self.h_pos, log_psi_q)
        V_neg, log_phi_hat, s2_neg = self._weighted_centroid(q, self.h_neg, log_phi_q)
        V = self._combine_drift(V_pos, V_neg, s2_pos, s2_neg)
        return V, log_psi_hat, log_phi_hat  # no tangential projection

    @torch.no_grad()
    def _abstain_score(self, q: Tensor, chunk: int = 64) -> Tensor:
        d = torch.cdist(q, self.h_neg)
        k = min(self.abstain_k, self.h_neg.size(0))
        return d.topk(k, dim=1, largest=False).values[:, -1]

    def vector_field(self, X: Tensor) -> Tensor:
        assert self.h_pos is not None
        self._to(X.device, X.dtype)
        V = self._field(X)[0]
        return V / (V.norm(dim=-1, keepdim=True) + _EPS)

    @torch.no_grad()
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        if self.h_pos is None or T == 0.0:
            return X
        R = self.R
        self._to(X.device, X.dtype)
        X_norm = X.norm(dim=-1, keepdim=True).clamp(min=_SAME_POINT_THR)
        # the two gates are scale free and read the *direction* of X, so they are literally
        # the same function of X in both variants; only the update geometry differs
        p0_dir = X * (R / X_norm)

        strength, active = self._compute_strength(p0_dir)
        if self.abstain_percentile is not None and self.rho_ref is not None:
            if self._gate_cache is None:
                self._gate_cache = self._abstain_gate(X)
            strength = strength * self._gate_cache
            active = strength > 0
        cos_T0 = ((p0_dir / R) * self.mu_T).sum(-1).clamp(-1 + _EPS, 1 - _EPS)
        theta_0 = torch.acos(cos_T0)
        dt_vec = (T * strength * theta_0 * R) / self.max_iters  # same arc length budget

        q = X.clone()
        for _ in range(self.max_iters):
            if not active.any():
                break
            V = self._field(q)[0]
            v_norm = V.norm(dim=-1, keepdim=True).clamp(min=_EPS)
            step = dt_vec.unsqueeze(-1) * (V / v_norm)
            q = torch.where(active.unsqueeze(-1), q + step, q)
        return torch.where(active.unsqueeze(-1), q, X)  # no radial rescale
