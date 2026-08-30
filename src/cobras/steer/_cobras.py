from __future__ import annotations

import torch
from torch import Tensor

from ._base_steer import Steer
from ..utils.sphere import exp_map, knn_geodesic_dist


_EPS = 1e-7
_SAME_POINT_THR = 1e-8


class COBRAS(Steer):
    """Conditional Optimal Bridge for Riemannian Activation Steering.

    This class is the shipped method and nothing else: the entropic transport plan of
    Eq. (7), the extended potentials of Eqs. (15)-(16), the Riemannian centroid drift of
    Eq. (19), the unit-length geodesic Euler step, the von Mises-Fisher strength, and the
    time-marginal abstention gate. Every knob that exists only to *disable* one of those parts
    lives in `AblationCOBRAS` (`_cobras_ablation.py`), which overrides the small hook
    methods below. Keeping the two apart means the defaults here are the paper's
    configuration by construction rather than by convention.

    Hook methods a variant may override: `_bandwidth`, `_kernel_weights`, `_potentials`,
    `_combine_drift`, `_abstain_score`, `_abstain_reference`, `_abstain_gate`, `_step`.
    """

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
        abstain_bandwidth_scale: float = 1.0 / 256.0,
        abstain_on_queries: bool = False,
    ) -> None:
        super().__init__()
        self.k_bw = int(k_bw)
        self.n_sinkhorn = int(n_sinkhorn)
        self.alpha_sigma = float(alpha_sigma)
        self.epsilon = float(epsilon)
        self.max_iters = int(max_iters)
        # None disables the vMF strength, i.e. every query is steered at full strength
        self.vmf_kappa = None if vmf_kappa is None else float(vmf_kappa)
        self.vmf_beta = float(vmf_beta)
        # nominal in-distribution coverage. 1.0 means "open everywhere", i.e. no gate, which
        # is also the historical spelling and is kept so existing configs keep working
        self.abstain_percentile = (
            None if abstain_percentile is None or abstain_percentile == 1.0
            else float(abstain_percentile)
        )
        self.abstain_bandwidth_scale = float(abstain_bandwidth_scale)
        self.abstain_on_queries = bool(abstain_on_queries)

        self.R: float | None = None
        self.abstain_ref: Tensor | None = None
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

    # ---------------------------------------------------------------- fitting

    @torch.no_grad()
    def fit(self, pos_X: Tensor, neg_X: Tensor, ref_X: Tensor | None = None) -> "COBRAS":
        """Solve the entropic bridge between the contrastive sets.

        `ref_X` holds in-distribution *query* activations and is used only to calibrate
        the abstention threshold, and only when `abstain_on_queries` is set. It matters
        because the contrastive pairs are answer activations while the gate is applied to
        prompt activations, which sit at a different position in the prompt and so at a
        systematically different radius; calibrating on one and thresholding the other
        shifts the whole reference. `scripts/prepare/extract_query_activations.py` writes
        these tensors.
        """
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

        diff = self.h_pos.mean(0) - self.h_neg.mean(0)
        self.mu_T = diff / diff.norm().clamp(min=_EPS)
        self._device = pos.device

        if self.abstain_percentile is not None:
            self.abstain_ref = self._abstain_reference(H_all, ref_X).sort().values

        self._post_fit(H_all)
        return self

    def _abstain_reference(self, H_all: Tensor, ref_X: Tensor | None) -> Tensor:
        """The sample of scores the coverage quantile is read off.

        Scoring the fitted set measures the marginal where the bridge put its own mass, which
        is systematically denser than a held-out query sits; scoring `ref_X` measures it where
        the gate is actually applied. The two are deliberately not interchangeable.
        """
        if not self.abstain_on_queries:
            return self._abstain_score(H_all)
        if ref_X is None:
            raise ValueError(
                "abstain_on_queries=True requires ref_X, but none was passed. Generate the "
                "query activations with scripts/prepare/extract_query_activations.py, or set "
                "abstain_on_queries=False to calibrate on the contrastive negatives instead. "
                "Falling back silently would change the gate threshold without changing the "
                "run name."
            )
        q_ref = ref_X.detach().to(dtype=self.h_pos.dtype, device=self.h_pos.device)
        q_ref = q_ref * (self.R / q_ref.norm(dim=-1, keepdim=True).clamp(min=_SAME_POINT_THR))
        return self._abstain_score(q_ref)

    def _post_fit(self, H_all: Tensor) -> None:
        """Hook for variants that need extra calibration after the bridge is solved."""

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

    # ------------------------------------------------------------ the bridge

    def _bandwidth(self, dist2: Tensor) -> Tensor | float:
        """Query-adaptive KDE bandwidth of Eq. (18): the squared geodesic distance to the
        furthest sample, so the weights are scale-free in how far the query has drifted."""
        return dist2.max(dim=-1, keepdim=True).values.clamp(min=(self.R * 1e-3) ** 2)

    def _query_log_psi(self, q: Tensor) -> Tensor:
        """Per-query positive weights via marginalization over h_neg. Returns [B, N_pos]."""
        R = self.R
        cos_qn = (q @ self.h_neg.T) / (R ** 2)
        cos_qn = cos_qn.clamp(-1.0 + _EPS, 1.0 - _EPS)
        dist2_qn = (R * torch.acos(cos_qn)).pow(2)  # [B, N_neg]
        log_alpha = self.log_phi.unsqueeze(0) - dist2_qn / (2.0 * self._bandwidth(dist2_qn))
        # log_psi_q[b,i] = logsumexp_j(log_alpha[b,j] - cost[j,i])
        return torch.logsumexp(log_alpha.unsqueeze(2) - self.cost.unsqueeze(0), dim=1)

    def _query_log_phi(self, q: Tensor) -> Tensor:
        """Per-query negative weights via marginalization over h_pos. Returns [B, N_neg]."""
        R = self.R
        cos_qp = (q @ self.h_pos.T) / (R ** 2)
        cos_qp = cos_qp.clamp(-1.0 + _EPS, 1.0 - _EPS)
        dist2_qp = (R * torch.acos(cos_qp)).pow(2)  # [B, N_pos]
        log_beta = self.log_psi.unsqueeze(0) - dist2_qp / (2.0 * self._bandwidth(dist2_qp))
        # log_phi_q[b,j] = logsumexp_i(log_beta[b,i] - cost[j,i])
        return torch.logsumexp(log_beta.unsqueeze(1) - self.cost.unsqueeze(0), dim=2)

    def _potentials(self, q: Tensor) -> tuple[Tensor, Tensor]:
        """The Eqs. (15)-(16) extension: both potentials re-marginalized at the current
        iterate, so they follow the query as it moves along the geodesic."""
        return self._query_log_psi(q), self._query_log_phi(q)

    def _kernel_weights(self, q: Tensor, H: Tensor, log_w: Tensor):
        """The Eq. (18) weights at `q`, plus the geometry they were built from."""
        R = self.R
        cos_t = (q @ H.T) / (R ** 2)
        cos_t = cos_t.clamp(-1.0 + _EPS, 1.0 - _EPS)
        theta = torch.acos(cos_t)
        dist2 = (R * theta).pow(2)
        sigma2 = self._bandwidth(dist2)
        lw = log_w if log_w.dim() == 2 else log_w.unsqueeze(0)  # [B, N] or [1, N]
        log_wK = lw - dist2 / (2.0 * sigma2)
        log_Z = torch.logsumexp(log_wK, dim=-1)  # log of the Schroedinger potential at q
        wK = (log_wK - log_Z.unsqueeze(-1)).exp()
        return wK, theta, cos_t, log_Z, sigma2

    def _weighted_centroid(
        self, q: Tensor, H: Tensor, log_w: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | float]:
        wK, theta, cos_t, log_Z, sigma2 = self._kernel_weights(q, H, log_w)
        sin_t = torch.sin(theta).clamp(min=_SAME_POINT_THR)
        coeff = torch.where(theta < _SAME_POINT_THR, torch.zeros_like(theta), theta / sin_t)
        wKc = wK * coeff
        numer = wKc @ H - (wKc * cos_t).sum(-1, keepdim=True) * q
        return numer / (1.0 + self.alpha_sigma), log_Z, sigma2

    def _combine_drift(
        self, V_pos: Tensor, V_neg: Tensor, s2_pos: Tensor | float, s2_neg: Tensor | float
    ) -> Tensor:
        """Eq. (19): the difference of the two Riemannian centroids. The 1/sigma^2 of
        Eq. (31) is absorbed into the step size, which is what makes the step unit-length."""
        return V_pos - V_neg

    def _field(self, q: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (drift, log psi_hat, log phi_hat) at `q`, tangent to the sphere."""
        log_psi_q, log_phi_q = self._potentials(q)
        V_pos, log_psi_hat, s2_pos = self._weighted_centroid(q, self.h_pos, log_psi_q)
        V_neg, log_phi_hat, s2_neg = self._weighted_centroid(q, self.h_neg, log_phi_q)
        V = self._combine_drift(V_pos, V_neg, s2_pos, s2_neg)
        V = V - (V * q).sum(-1, keepdim=True) / (self.R ** 2) * q
        return V, log_psi_hat, log_phi_hat

    # -------------------------------------------------------- strength & gate

    def _compute_strength(self, p0: Tensor) -> tuple[Tensor, Tensor]:
        if self.vmf_kappa is None:
            strength = torch.ones(p0.shape[0], device=p0.device, dtype=p0.dtype)
            return strength, strength > 0
        cos_t = (p0 / self.R) @ self.mu_T
        logits = self.vmf_kappa * torch.stack([cos_t, -cos_t], -1)
        delta = torch.softmax(logits, -1)[:, 1]
        strength = ((delta - self.vmf_beta) / (1.0 - self.vmf_beta)).clamp(0.0, 1.0)
        return strength, strength > 0

    @torch.no_grad()
    def _abstain_score(self, q: Tensor, chunk: int = 128) -> Tensor:
        """How far a query sits from the transport support, read off the bridge itself.

        The probability-flow drift of Eq. (13) is the *ratio* of the two Schroedinger
        potentials, in which their common kernel decay in the distance to the data cancels
        exactly. The decay survives only in the *product*, which is the bridge's own time
        marginal p_0 = psi_hat * phi_hat (Sec. 4.2) -- the one factor of the SB solution the
        drift never reads, and so the only one that can say whether the bridge assigned this
        query any transport at all.

        The score is -log p_0 at bandwidth b = `abstain_bandwidth_scale` * sigma^2; larger
        means further out. The bandwidth picks a point in a one-parameter family the bridge
        already defines: -2b log phi_hat_b is the *mean* squared geodesic distance to the
        negatives at b = sigma^2, and converges to the *minimum* of that distance as b -> 0.
        The shipped default sits near the small-bandwidth end.
        """
        return -self.log_marginal(
            q, raw=True, bandwidth_scale=self.abstain_bandwidth_scale, chunk=chunk
        )

    def _abstain_gate(self, q: Tensor) -> Tensor:
        """Quantile calibration, and the gate's only knob.

        The score is mapped through its own empirical CDF under the reference sample, so
        `abstain_percentile` is a nominal in-distribution coverage rather than a threshold on
        a raw quantity: the gate is fully open for the most in-distribution fraction p of the
        reference and ramps linearly to zero over the rest. Under that reference the mean gate
        is exactly (1 + p) / 2, which is what makes p comparable across models and layers.
        """
        score = self._abstain_score(q)
        ref = self.abstain_ref.to(device=score.device, dtype=score.dtype)
        rank = torch.searchsorted(ref, score.contiguous()) / ref.numel()
        return ((1.0 - rank) / (1.0 - self.abstain_percentile)).clamp(0.0, 1.0)

    # ------------------------------------------------------------- steering

    def vector_field(self, X: Tensor) -> Tensor:
        assert self.h_pos is not None
        self._to(X.device, X.dtype)
        p = X * (self.R / X.norm(dim=-1, keepdim=True).clamp(min=_SAME_POINT_THR))
        V = self._field(p)[0]
        return V / (V.norm(dim=-1, keepdim=True) + _EPS)

    def _step(self, V: Tensor, dt_vec: Tensor, T: float, strength: Tensor) -> Tensor:
        """Unit-length geodesic Euler step: the drift sets the direction, `dt_vec` the
        distance, so total arc length is fixed by T regardless of the drift magnitude."""
        v_norm = V.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        return dt_vec.unsqueeze(-1) * (V / v_norm)

    @torch.no_grad()
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        if self.h_pos is None or T == 0.0:
            return X
        R = self.R
        self._to(X.device, X.dtype)
        X_norm = X.norm(dim=-1, keepdim=True).clamp(min=_SAME_POINT_THR)
        p0 = X * (R / X_norm)

        strength, active = self._compute_strength(p0)
        if self.abstain_percentile is not None and self.abstain_ref is not None:
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
            V = self._field(q)[0]
            step = self._step(V, dt_vec, T, strength)
            if self.epsilon > 0.0:
                v_norm = V.norm(dim=-1, keepdim=True).clamp(min=_EPS)
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

    # ------------------------------------------------------------- analysis

    def _geo_sq(self, q: Tensor, H: Tensor) -> Tensor:
        cos_t = ((q @ H.T) / (self.R ** 2)).clamp(-1.0 + _EPS, 1.0 - _EPS)
        return (self.R * torch.acos(cos_t)).pow(2)

    @torch.no_grad()
    def log_marginal(
        self, X: Tensor, raw: bool = False, bandwidth_scale: float = 1.0, chunk: int = 128
    ) -> Tensor:
        """log p_0 = log phi_hat + log psi_hat in the plain form of Eqs. (10)-(11), i.e. with
        the Sinkhorn potentials themselves rather than their Eq. (15)-(16) extension.

        The probability-flow drift is the *ratio* of the two potentials, in which their common
        kernel decay in the distance to the data cancels; that decay survives only in the
        *product*, which is the bridge's own time marginal (Sec. 4.2). So this score is the one
        factor of the SB solution the drift never reads, and it is what says whether the query
        lies where the bridge assigned any transport at all.

        `bandwidth_scale` multiplies sigma^2 here only. The scale matters: at b = sigma^2 the
        score is an affine function of the *mean* squared geodesic distance to the samples,
        while as b -> 0 it converges to the *minimum* of that distance.
        """
        self._to(X.device, X.dtype)
        p = X if raw else X * (self.R / X.norm(dim=-1, keepdim=True).clamp(min=_SAME_POINT_THR))
        b = bandwidth_scale * self.sigma2
        out = []
        for i in range(0, p.size(0), chunk):
            q = p[i : i + chunk]
            d2p = self._geo_sq(q, self.h_pos)
            d2n = self._geo_sq(q, self.h_neg)
            out.append(torch.logsumexp(self.log_phi.unsqueeze(0) - d2n / (2.0 * b), dim=-1)
                       + torch.logsumexp(self.log_psi.unsqueeze(0) - d2p / (2.0 * b), dim=-1))
        return torch.cat(out)

    @torch.no_grad()
    def attribution(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """The Eq. (18) weights at the query, i.e. the distribution over the contrastive
        samples whose convex combination is the step of Eq. (19). Returns (w_pos, w_neg),
        each summing to 1 along the sample axis, evaluated at the initial iterate q_0."""
        self._to(X.device, X.dtype)
        q0 = X * (self.R / X.norm(dim=-1, keepdim=True).clamp(min=_SAME_POINT_THR))
        log_psi_q, log_phi_q = self._potentials(q0)
        w_pos = self._kernel_weights(q0, self.h_pos, log_psi_q)[0]
        w_neg = self._kernel_weights(q0, self.h_neg, log_phi_q)[0]
        return w_pos, w_neg
