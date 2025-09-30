"""
Filename: gate_specs.py
Author: Alexay Mehra
Date: 2025-09-28
Description: 
"""


# Imports
import numpy as np


# Class for gate building ----------------------------------------------------------------
# ----------------------------------------------------------------------------------------
class GateSpec:
    def __init__(self, name, n_params, builder, config=None):
        self.name = name                # name of the gate
        self.n_params = n_params        # number of optimizable parameters
        self.builder = builder          # function: (params, backend, config) -> matrix
        self.config = config or {}      # optional extra parameters for the gate (not touched by optimization)

    def build(self, params, backend):
        if len(params) != self.n_params:
            raise ValueError(f"{self.name} expects {self.n_params} params, got {len(params)}")
        return self.builder(params, backend, self.config)
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------


# Specific gates that can be built -------------------------------------------------------
# ----------------------------------------------------------------------------------------
PhaseSpaceRotationGate = GateSpec(
    "Phase Space Rotation",
    1,
    lambda params, backend, config: backend.full_phase_space_rotation(params[0])
)

DisplacementGate = GateSpec(
    "Displacement",
    2,
    lambda params, backend, config: backend.full_displacement(params[0] + 1j*params[1])
)

SqueezingGate = GateSpec(
    "Squeezing",
    1,
    lambda params, backend, config: backend.full_squeezing(params[0])
)

RotationGate = GateSpec(
    "XY Rotation",
    2,
    lambda params, backend, config: backend.full_qubit_xy_rotation(params[0], params[1])
)


AlwaysOnGate = GateSpec(
    "Always-On Evolution",
    n_params = 0,                                       # optimization doesn't touch any parameters
    builder = lambda params, backend, config: backend.always_on_evolution(
        omega = config.get("omega", 1),
        chi = config.get("chi", 0.01),
        t = config.get("time", 1)
    ),
    config = {"omega": 1, "chi": 0.01, "time": 1}       # allows for notebook-configurable parameters
)
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
