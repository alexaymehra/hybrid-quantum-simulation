"""
Filename: gate_specs.py
Author: Alexay Mehra
Date: 2025-09-28
Description: Holds the Class for building gates, and specific gates for optimizaiton.
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
ControlledRotation = GateSpec(
    "Controlled Rotation",
    n_params = 0,                                       # optimization doesn't touch any parameters
    builder = lambda params, backend, config: backend.c_r(
        omega = config.get("omega", 1),
        chi = config.get("chi", 0.01),
        t = config.get("time", 1)
    ),
    config = {"omega": 1, "chi": 0.01, "time": 1}       # allows for notebook-configurable parameters
)

QumodeRotation = GateSpec(
    "Phase Space Rotation",
    1,
    lambda params, backend, config: backend.cv_r_full(params[0])
)

Displacement = GateSpec(
    "Displacement",
    2,
    lambda params, backend, config: backend.cv_d_full(params[0] + 1j*params[1])
)

Squeezing = GateSpec(
    "Squeezing",
    1,
    lambda params, backend, config: backend.cv_s_full(params[0])
)

QubitRotation = GateSpec(
    "Qubit XY Rotation",
    2,
    lambda params, backend, config: backend.q_xy_r_full(params[0], params[1])
)

ControlledDisplacement = GateSpec(
    "Controlled Displacement",
    1,
    lambda params, backend, config: backend.c_d(params[0])
)
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
