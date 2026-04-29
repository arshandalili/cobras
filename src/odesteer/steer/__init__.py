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

#COBRAS
from ._cobras import COBRAS


__all__ = [
    'Steer', 'VecSteer',
    # Baselines
    'RepE', 'CAA', 'ITI', 'MiMiC', 'LinAcT', 'SphericalSteer',
    # ODESteer
    'BaseODESteer', 'ODESteer', 'RFFODESteer',
    'BaseStepODESteer', 'StepODESteer', 'RFFStepODESteer',
    # COBRAS
    'COBRAS',
]

def get_steer_model(name: str, *args, **kwargs) -> type[Steer]:
    if name == "NoSteer":
        return None
    return getattr(sys.modules[__name__], name)(*args, **kwargs)