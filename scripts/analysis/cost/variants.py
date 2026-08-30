"""Q4 (Reviewer EJdD): approximations to COBRAS that cut the cost at scale.

Standalone; nothing under src/ is modified.

1. subsample_fit(...)      -- fit the bridge on a random fraction of D^+/D^-.
                             No code change needed, only what is passed to fit().

2. TopKCOBRAS(topk=k)      -- online truncation. Eq. (18) is a softmax over all
                             N samples, so restricting it to the k nearest
                             contrastive samples of the current iterate is a
                             truncation of that softmax with controllable error.
                             The same k-neighbourhood also truncates the
                             marginalisation of Eqs. (15)-(16), which is what
                             turns the O(B N_+ N_-) per-step term into O(B k^2).

                             The Eq. (18) bandwidth sigma^2(x) = max_i d^2 is
                             kept at its full-set value (the full distance
                             vector is computed anyway for the neighbour search)
                             so that the truncated weights are exactly the
                             full-N weights restricted and renormalised.

3. max_iters=K             -- fewer geodesic Euler steps. Already a constructor
                             argument of COBRAS.

Per-step cost, batch B, width d, K Euler steps, per generated token:
  COBRAS      K * ( O(B (N_+ + N_-) d)  +  O(B N_+ N_-) )
  TopKCOBRAS  K * ( O(B (N_+ + N_-) d)  +  O(B k^2) + O(B k d) )
"""
from __future__ import annotations

import torch
from torch import Tensor

# the ablation switches this script sweeps (potentials=, drift=, bandwidth=, step_mode=,
# uniform_weights=, abstain_signal=) live on AblationCOBRAS; with defaults it is COBRAS
from cobras.steer import AblationCOBRAS as COBRAS

_EPS = 1e-7
_SAME_POINT_THR = 1e-8


def subsample(X: Tensor, frac: float, seed: int = 0) -> Tensor:
    """Random subset of the rows of X, keeping ceil(frac * N) of them."""
    if frac >= 1.0:
        return X
    n = max(2, int(round(frac * X.size(0))))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.size(0), generator=g)[:n]
    return X[idx.to(X.device)]


class TopKCOBRAS(COBRAS):
    """COBRAS whose Eq. (15)-(19) evaluation is restricted, at every Euler step,
    to the `topk` geodesically nearest positives and negatives of the current
    iterate. `topk=None` recovers the exact field."""

    def __init__(self, topk: int | None = 64, **kwargs):
        super().__init__(**kwargs)
        self.topk = None if topk is None else int(topk)

    def _field(self, q: Tensor):
        if self.topk is None:
            return super()._field(q)
        R = self.R
        kp = min(self.topk, self.h_pos.size(0))
        kn = min(self.topk, self.h_neg.size(0))

        # full distance vectors: needed for the neighbour search, and they give
        # the full-set adaptive bandwidth for free
        cos_qp = ((q @ self.h_pos.T) / (R ** 2)).clamp(-1.0 + _EPS, 1.0 - _EPS)
        cos_qn = ((q @ self.h_neg.T) / (R ** 2)).clamp(-1.0 + _EPS, 1.0 - _EPS)
        th_p_all, th_n_all = torch.acos(cos_qp), torch.acos(cos_qn)
        d2p_all, d2n_all = (R * th_p_all).pow(2), (R * th_n_all).pow(2)
        s2p = self._bandwidth(d2p_all)           # [B,1] adaptive, or scalar if fixed
        s2n = self._bandwidth(d2n_all)

        ip = d2p_all.topk(kp, dim=-1, largest=False).indices      # [B, kp]
        i_n = d2n_all.topk(kn, dim=-1, largest=False).indices     # [B, kn]

        d2p = d2p_all.gather(1, ip)
        d2n = d2n_all.gather(1, i_n)
        th_p = th_p_all.gather(1, ip)
        th_n = th_n_all.gather(1, i_n)
        cos_p = cos_qp.gather(1, ip)
        cos_n = cos_qn.gather(1, i_n)
        Hp = self.h_pos[ip]                                       # [B, kp, d]
        Hn = self.h_neg[i_n]                                      # [B, kn, d]

        if self.potentials == "plain":
            log_psi_q = self.log_psi[ip]
            log_phi_q = self.log_phi[i_n]
        else:
            # cost sub-block C[b] = cost[i_n[b]][:, ip[b]] -> [B, kn, kp]
            C = self.cost[i_n.unsqueeze(-1), ip.unsqueeze(1)]
            log_alpha = self.log_phi[i_n] - d2n / (2.0 * s2n)     # [B, kn]
            log_psi_q = torch.logsumexp(log_alpha.unsqueeze(2) - C, dim=1)   # [B, kp]
            log_beta = self.log_psi[ip] - d2p / (2.0 * s2p)       # [B, kp]
            log_phi_q = torch.logsumexp(log_beta.unsqueeze(1) - C, dim=2)    # [B, kn]

        V_pos, lp = self._centroid_sub(q, Hp, log_psi_q, d2p, th_p, cos_p, s2p)
        V_neg, ln = self._centroid_sub(q, Hn, log_phi_q, d2n, th_n, cos_n, s2n)
        V = (V_pos / s2p - V_neg / s2n) if self.drift == "gradient" else (V_pos - V_neg)
        V = V - (V * q).sum(-1, keepdim=True) / (R ** 2) * q
        return V, lp, ln

    def _centroid_sub(self, q, H, log_w, dist2, theta, cos_t, sigma2):
        """_weighted_centroid restricted to a per-query neighbourhood H [B, k, d]."""
        log_wK = log_w - dist2 / (2.0 * sigma2)
        log_Z = torch.logsumexp(log_wK, dim=-1)
        wK = (log_wK - log_Z.unsqueeze(-1)).exp()
        if self.uniform_weights:
            wK = torch.full_like(wK, 1.0 / wK.shape[-1])
        sin_t = torch.sin(theta).clamp(min=_SAME_POINT_THR)
        coeff = torch.where(theta < _SAME_POINT_THR, torch.zeros_like(theta), theta / sin_t)
        wKc = wK * coeff
        numer = torch.bmm(wKc.unsqueeze(1), H).squeeze(1) - (wKc * cos_t).sum(-1, keepdim=True) * q
        return numer / (1.0 + self.alpha_sigma), log_Z
