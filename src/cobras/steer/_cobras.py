from __future__ import annotations

import torch
from torch import Tensor

from ._base_steer import Steer
from ..utils.sphere import exp_map, knn_geodesic_dist


_EPS = 1e-7
_SAME_POINT_THR = 1e-8


class COBRAS(Steer):
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
    ) -> None:
        super().__init__()
        self.k_bw = int(k_bw)
        self.n_sinkhorn = int(n_sinkhorn)
        self.alpha_sigma = float(alpha_sigma)
        self.epsilon = float(epsilon)
        self.max_iters = int(max_iters)
        self.vmf_kappa = float(vmf_kappa)
        self.vmf_beta = float(vmf_beta)
        self.abstain_percentile = None if (abstain_percentile==1.0) else float(abstain_percentile)
        self.abstain_k = int(abstain_k)
        self.abstain_sharpness = float(abstain_sharpness)

        self.R: float | None = None
        self.rho_ref: float | None = None
        self.sigma2: float | None = None
        self.h_pos: Tensor | None = None
        self.h_neg: Tensor | None = None
        self.mu_T: Tensor | None = None
        self.log_psi: Tensor | None = None
        self.log_phi: Tensor | None = None
        self.cost: Tensor | None = None
        self._device: torch.device | None = None
        self._gate_cache: Tensor | None = None

    def reset_gate(self) -> None:
        self._gate_cache = None

    @torch.no_grad()
    def fit(self, pos_X: Tensor, neg_X: Tensor) -> "COBRAS":
        pos = pos_X.detach().to(torch.float32)
        neg = neg_X.detach().to(torch.float32)
        R = torch.cat([pos, neg], 0).norm(dim=-1).mean().item()
        self.R = float(R)
        self.h_pos = pos * (R / pos.norm(dim=-1, keepdim=True).clamp(min=_SAME_POINT_THR))
        self.h_neg = neg * (R / neg.norm(dim=-1, keepdim=True).clamp(min=_SAME_POINT_THR))

        H_all = torch.cat([self.h_pos, self.h_neg], 0)
        k = min(self.k_bw, H_all.size(0) - 1)
        sigma2 = knn_geodesic_dist(H_all, R, k=k).median().item() ** 2
        self.sigma2 = float(max(sigma2, (R * 1e-3) ** 2))

        cos_np = (self.h_neg @ self.h_pos.T) / (R ** 2)
        cos_np = cos_np.clamp(-1.0 + _EPS, 1.0 - _EPS)
        cost = (R * torch.acos(cos_np)).pow(2) / (2.0 * self.sigma2)
        self.cost = cost
        self.log_psi, self.log_phi = self._sinkhorn(cost, self.n_sinkhorn)

        if self.abstain_percentile is not None:
            neg_rho = knn_geodesic_dist(self.h_neg, R, k=min(self.abstain_k, self.h_neg.size(0) - 1))
            self.rho_ref = float(torch.quantile(neg_rho.float(), self.abstain_percentile).item())

        diff = self.h_pos.mean(0) - self.h_neg.mean(0)
        self.mu_T = diff / diff.norm().clamp(min=_EPS)
        self._device = pos.device
        return self

    @staticmethod
    def _sinkhorn(cost: Tensor, n_iters: int) -> tuple[Tensor, Tensor]:
        N_neg, N_pos = cost.shape
        device, dtype = cost.device, cost.dtype
        log_p_neg = -torch.log(torch.tensor(float(N_neg), device=device, dtype=dtype))
        log_p_pos = -torch.log(torch.tensor(float(N_pos), device=device, dtype=dtype))
        log_psi = torch.zeros(N_pos, device=device, dtype=dtype)
        log_phi = torch.zeros(N_neg, device=device, dtype=dtype)
        for _ in range(n_iters):
            log_phi = log_p_neg - torch.logsumexp(log_psi[None, :] - cost, dim=1)
            log_psi = log_p_pos - torch.logsumexp(log_phi[:, None] - cost, dim=0)
        return log_psi, log_phi

    def _to(self, device: torch.device, dtype: torch.dtype) -> None:
        if self._device == device and self.h_pos.dtype == dtype:
            return
        for name in ("h_pos", "h_neg", "mu_T", "log_psi", "log_phi", "cost"):
            t = getattr(self, name, None)
            if t is not None:
                setattr(self, name, t.to(device=device, dtype=dtype))
        self._device = device

    def _query_log_psi(self, q: Tensor) -> Tensor:
        """Per-query positive weights via marginalization over h_neg. Returns [B, N_pos]."""
        R = self.R
        cos_qn = (q @ self.h_neg.T) / (R ** 2)
        cos_qn = cos_qn.clamp(-1.0 + _EPS, 1.0 - _EPS)
        dist2_qn = (R * torch.acos(cos_qn)).pow(2)  # [B, N_neg]
        sigma2_qn = dist2_qn.max(dim=-1, keepdim=True).values.clamp(min=(R * 1e-3) ** 2)
        log_alpha = self.log_phi.unsqueeze(0) - dist2_qn / (2.0 * sigma2_qn)  # [B, N_neg]
        # log_psi_q[b,i] = logsumexp_j(log_alpha[b,j] - cost[j,i])
        return torch.logsumexp(log_alpha.unsqueeze(2) - self.cost.unsqueeze(0), dim=1)  # [B, N_pos]

    def _query_log_phi(self, q: Tensor) -> Tensor:
        """Per-query negative weights via marginalization over h_pos. Returns [B, N_neg]."""
        R = self.R
        cos_qp = (q @ self.h_pos.T) / (R ** 2)
        cos_qp = cos_qp.clamp(-1.0 + _EPS, 1.0 - _EPS)
        dist2_qp = (R * torch.acos(cos_qp)).pow(2)  # [B, N_pos]
        sigma2_qp = dist2_qp.max(dim=-1, keepdim=True).values.clamp(min=(R * 1e-3) ** 2)
        log_beta = self.log_psi.unsqueeze(0) - dist2_qp / (2.0 * sigma2_qp)  # [B, N_pos]
        # log_phi_q[b,j] = logsumexp_i(log_beta[b,i] - cost[j,i])
        return torch.logsumexp(log_beta.unsqueeze(1) - self.cost.unsqueeze(0), dim=2)  # [B, N_neg]

    def _weighted_centroid(self, q: Tensor, H: Tensor, log_w: Tensor) -> Tensor:
        R = self.R
        cos_t = (q @ H.T) / (R ** 2)
        cos_t = cos_t.clamp(-1.0 + _EPS, 1.0 - _EPS)
        theta = torch.acos(cos_t)
        dist2 = (R * theta).pow(2)
        sigma2 = dist2.max(dim=-1, keepdim=True).values.clamp(min=(R * 1e-3) ** 2)
        lw = log_w if log_w.dim() == 2 else log_w.unsqueeze(0)  # [B, N] or [1, N]
        log_wK = lw - dist2 / (2.0 * sigma2)
        wK = (log_wK - torch.logsumexp(log_wK, dim=-1, keepdim=True)).exp()
        sin_t = torch.sin(theta).clamp(min=_SAME_POINT_THR)
        coeff = torch.where(theta < _SAME_POINT_THR, torch.zeros_like(theta), theta / sin_t)
        wKc = wK * coeff
        numer = wKc @ H - (wKc * cos_t).sum(-1, keepdim=True) * q
        return numer / (1.0 + self.alpha_sigma)

    def _field(self, q: Tensor) -> Tensor:
        R = self.R
        log_psi_q = self._query_log_psi(q)  # [B, N_pos] — adapts to current q
        log_phi_q = self._query_log_phi(q)   # [B, N_neg] — symmetric, adapts to current q
        V_pos = self._weighted_centroid(q, self.h_pos, log_psi_q)
        V_neg = self._weighted_centroid(q, self.h_neg, log_phi_q)
        V = V_pos - V_neg
        return V - (V * q).sum(-1, keepdim=True) / (R ** 2) * q

    def _compute_strength(self, p0: Tensor) -> tuple[Tensor | None, Tensor]:
        cos_t = (p0 / self.R) @ self.mu_T
        logits = self.vmf_kappa * torch.stack([cos_t, -cos_t], -1)
        delta = torch.softmax(logits, -1)[:, 1]
        strength = ((delta - self.vmf_beta) / (1.0 - self.vmf_beta)).clamp(0.0, 1.0)
        return strength, strength > 0

    def _abstain_gate(self, q: Tensor) -> Tensor:
        cos_qn = (q @ self.h_neg.T) / (self.R ** 2)
        cos_qn = cos_qn.clamp(-1.0 + _EPS, 1.0 - _EPS)
        dist_qn = self.R * torch.acos(cos_qn)
        k = min(self.abstain_k, self.h_neg.size(0))
        rho = dist_qn.topk(k, dim=1, largest=False).values[:, -1]
        return 1.0 / (1.0 + (rho / self.rho_ref) ** self.abstain_sharpness)

    def vector_field(self, X: Tensor) -> Tensor:
        assert self.h_pos is not None
        self._to(X.device, X.dtype)
        p = X * (self.R / X.norm(dim=-1, keepdim=True).clamp(min=_SAME_POINT_THR))
        V = self._field(p)
        return V / (V.norm(dim=-1, keepdim=True) + _EPS)

    @torch.no_grad()
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        if self.h_pos is None or T == 0.0:
            return X
        R = self.R
        self._to(X.device, X.dtype)
        X_norm = X.norm(dim=-1, keepdim=True).clamp(min=_SAME_POINT_THR)
        p0 = X * (R / X_norm)

        strength, active = self._compute_strength(p0)
        if self.abstain_percentile is not None and self.rho_ref is not None:
            if self._gate_cache is None:
                self._gate_cache = self._abstain_gate(p0)
            strength = strength * self._gate_cache
            active = strength > 0
        cos_T0 = ((p0 / R) * self.mu_T).sum(-1).clamp(-1 + _EPS, 1 - _EPS)
        theta_0 = torch.acos(cos_T0)
        dt_vec = (T * strength * theta_0 * R) / self.max_iters

        q = p0.clone()
        for _ in range(self.max_iters):
            if not active.any():
                break
            V = self._field(q)
            v_norm = V.norm(dim=-1, keepdim=True).clamp(min=_EPS)
            step = dt_vec.unsqueeze(-1) * (V / v_norm)
            if self.epsilon > 0.0:
                xi = torch.randn_like(q)
                q_unit = q / R
                xi = xi - (xi * q_unit).sum(-1, keepdim=True) * q_unit
                v_hat = V / v_norm
                xi = xi - (xi * v_hat).sum(-1, keepdim=True) * v_hat
                xi = xi / xi.norm(dim=-1, keepdim=True).clamp(min=_SAME_POINT_THR)
                step = step + (2.0 * self.epsilon * T / self.max_iters) ** 0.5 * R * xi
            q_new = exp_map(q, step, R)
            q = torch.where(active.unsqueeze(-1), q_new, q)

        return torch.where(active.unsqueeze(-1), q, p0) * (X_norm / R)
