"""
Filename: gate_specs.py
Author: Alexay Mehra
Date: 2025-09-28
Description: 
"""

import numpy as np


# Class for gate building ----------------------------------------------------------------
# ----------------------------------------------------------------------------------------
class GateSpec:
    def __init__(self, name, n_params, builder, config=None):
        self.name = name
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
DisplacementGate = GateSpec(
    "Displacement",
    2,
    lambda params, backend: backend.full_displacement(params[0] + 1j*params[1])
)


RotationGate = GateSpec(
    "XY Rotation",
    2,
    lambda params, backend: backend.full_qubit_xy_rotation(params[0], params[1])
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


# Building a generic gate sequence -------------------------------------------------------
# ----------------------------------------------------------------------------------------
"""
Builds the full gate sequence for d layers:

params       : flat array of optimization parameters
seq_template : list of GateSpec objects representing a single layer
backend      : backend object with operators
d            : number of times to repeat the layer
"""
def build_sequence(params, seq_template, backend, d=1):
    tot_dim = backend.I_o.shape[0] * backend.I_q.shape[0]   # total hilbert space dimension
    U = np.eye(tot_dim, dtype=complex)                      # starting point for the sequence
    idx = 0                                                 # keeps track of position in params

    full_seq = seq_template * d                             # repeat the sequence d times

    for gate in full_seq:                                   # loop through all gates in the sequence
        gate_params = params[idx: idx + gate.n_params]      # slice out parameters for the current gate
        U = U @ gate.build(gate_params, backend)            # multiply the gate into the sequence
        idx += gate.n_params                                # increment idx

    return U                                                # return full sequence
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------