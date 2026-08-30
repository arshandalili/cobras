"""The invariant that keeps the ablation split honest.

`AblationCOBRAS` exists so `COBRAS` can stay the shipped method. That only holds if
`AblationCOBRAS` with default arguments *is* `COBRAS` -- every switch's default must be the
shipped behaviour. If someone adds a knob whose default changes the path, this fails.
"""

import pytest
import torch

from cobras.steer import COBRAS, AblationCOBRAS, EuclideanCOBRAS


SHIPPED = dict(k_bw=5, n_sinkhorn=5, alpha_sigma=1e-3, epsilon=0.0, max_iters=10,
               vmf_kappa=20, vmf_beta=0.0)


@pytest.fixture
def data():
    torch.manual_seed(0)
    d = 64
    return (torch.randn(40, d) * 3 + 1.0,
            torch.randn(40, d) * 3 - 1.0,
            torch.randn(25, d) * 3)


@pytest.mark.parametrize("gate", [1.0, 0.98])
def test_ablation_defaults_are_the_shipped_method(data, gate):
    pos, neg, q = data
    kw = {**SHIPPED, "abstain_percentile": gate, "abstain_sharpness": 200.0}
    a = COBRAS(**kw).fit(pos.clone(), neg.clone()).steer(q.clone(), T=0.65)
    b = AblationCOBRAS(**kw).fit(pos.clone(), neg.clone()).steer(q.clone(), T=0.65)
    assert torch.equal(a, b)


def test_gate_on_queries_refuses_to_fall_back_silently(data):
    """Calibrating on the negatives instead of on real queries moves the threshold without
    changing the run name, so a missing ref_X has to be an error rather than a default."""
    pos, neg, _ = data
    m = COBRAS(**SHIPPED, abstain_percentile=0.98, abstain_on_queries=True)
    with pytest.raises(ValueError, match="requires ref_X"):
        m.fit(pos, neg)


def test_euclidean_control_ignores_the_sphere(data):
    """EuclideanCOBRAS keeps norms, so its output must not lie on the fitted radius."""
    pos, neg, q = data
    m = EuclideanCOBRAS(**SHIPPED, abstain_percentile=1.0).fit(pos.clone(), neg.clone())
    out = m.steer(q.clone(), T=0.65)
    assert out.shape == q.shape
    assert not torch.allclose(out.norm(dim=-1), q.norm(dim=-1))


def test_unknown_steer_model_names_itself(data):
    from cobras.steer import get_steer_model
    with pytest.raises(ValueError, match="unknown steer model"):
        get_steer_model("COBRAZ")
