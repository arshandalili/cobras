import sys
from ._base_steer import Steer

# Vector-based Steers
from ._repe import RepE
from ._caa import CAA
from ._iti import ITI

# OT-based Steers
from ._mimic import MiMiC
from ._lin_act import LinAcT

# ODESteer
from ._ode_steer import BaseODESteer, ODESteer, RFFODESteer
from ._step_ode_steer import BaseStepODESteer, StepODESteer, RFFStepODESteer

# Spherical Steer
from ._spherical_steer import SphericalSteer

# COBRAS: the shipped method, its ablation variants, and its Euclidean control
from ._cobras import COBRAS
from ._cobras_ablation import AblationCOBRAS
from ._euclidean_cobras import EuclideanCOBRAS

# Wraps any steering method in COBRAS's abstention gate, so the gate's contribution
# can be separated from the steering rule's
from ._gated_steer import GatedSteer


__all__ = [
    'Steer', 'VecSteer', 'GatedSteer',
    # Baselines
    'RepE', 'CAA', 'ITI', 'MiMiC', 'LinAcT', 'SphericalSteer',
    # ODESteer
    'BaseODESteer', 'ODESteer', 'RFFODESteer',
    'BaseStepODESteer', 'StepODESteer', 'RFFStepODESteer',
    # COBRAS
    'COBRAS', 'AblationCOBRAS', 'EuclideanCOBRAS',
]

def get_steer_model(name: str, *args, **kwargs) -> type[Steer]:
    if name == "NoSteer":
        return None
    try:
        cls = getattr(sys.modules[__name__], name)
    except AttributeError:
        raise ValueError(
            f"unknown steer model {name!r}; available: {', '.join(sorted(__all__))}"
        ) from None
    return cls(*args, **kwargs)
