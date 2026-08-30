'''
Wraps any steering method with the kNN abstention gate of COBRAS, so that the OOD
contribution of the gate can be separated from that of the steering rule itself.
'''

import torch
from torch import Tensor

from ._base_steer import Steer
from ..utils.sphere import knn_geodesic_dist


_EPS = 1e-7
_SAME_POINT_THR = 1e-8


class GatedSteer(Steer):
    def __init__(
        self,
        base: str,
        abstain_percentile: float = 0.98,
        abstain_k: int = 32,
        abstain_sharpness: float = 50.0,
        abstain_on_queries: bool = False,
        base_kwargs: dict = {},
    ) -> None:
        super().__init__()
        from . import get_steer_model
        self.base = get_steer_model(base, **base_kwargs)
        self.abstain_percentile = float(abstain_percentile)
        self.abstain_k = int(abstain_k)
        self.abstain_sharpness = float(abstain_sharpness)
        self.abstain_on_queries = bool(abstain_on_queries)

        self.R: float | None = None
        self.rho_ref: float | None = None
        self.h_neg: Tensor | None = None
        self._gate_cache: Tensor | None = None

    def reset_gate(self) -> None:
        self._gate_cache = None
        if hasattr(self.base, 'reset_gate'):
            self.base.reset_gate()

    @torch.no_grad()
    def fit(self, pos_X: Tensor, neg_X: Tensor, ref_X: Tensor | None = None) -> 'GatedSteer':
        self.base.fit(pos_X, neg_X)
        pos = pos_X.detach().to(torch.float32)
        neg = neg_X.detach().to(torch.float32)
        R = torch.cat([pos, neg], 0).norm(dim = -1).mean().item()
        self.R = float(R)
        self.h_neg = neg * (R / neg.norm(dim = -1, keepdim = True).clamp(min = _SAME_POINT_THR))
        if ref_X is None or not self.abstain_on_queries:
            rho = knn_geodesic_dist(self.h_neg, R, k = min(self.abstain_k, self.h_neg.size(0) - 1))
        else:
            rho = self._radius(ref_X.detach().to(torch.float32))
        self.rho_ref = float(torch.quantile(rho.float(), self.abstain_percentile).item())
        return self

    def _radius(self, X: Tensor) -> Tensor:
        p = X * (self.R / X.norm(dim = -1, keepdim = True).clamp(min = _SAME_POINT_THR))
        cos_qn = ((p @ self.h_neg.T) / (self.R ** 2)).clamp(-1.0 + _EPS, 1.0 - _EPS)
        dist_qn = self.R * torch.acos(cos_qn)
        k = min(self.abstain_k, self.h_neg.size(0))
        return dist_qn.topk(k, dim = 1, largest = False).values[:, -1]

    def gate(self, X: Tensor) -> Tensor:
        self.h_neg = self.h_neg.to(device = X.device, dtype = X.dtype)
        rho = self._radius(X)
        return 1.0 / (1.0 + (rho / self.rho_ref) ** self.abstain_sharpness)

    @torch.no_grad()
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        if self._gate_cache is None:
            self._gate_cache = self.gate(X)
        return X + self._gate_cache.unsqueeze(-1) * (self.base.steer(X, T) - X)

    def vector_field(self, X: Tensor) -> Tensor:
        return self.base.vector_field(X)
