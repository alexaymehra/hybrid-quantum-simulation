"""
Filename: gate_specs.py
Author: Alexay Mehra
Date: 2025-09-28
Description: 
"""


# Imports
import numpy as np


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


# Cost (Infidelity) Function -------------------------------------------------------------
# ----------------------------------------------------------------------------------------
def fidelity_loss(params, seq_template, backend, d, U_target):
    U = build_sequence(params, seq_template, backend, d)
    dim = U.shape[0]
    fid = np.abs(np.trace(U.conj().T @ U_target))/ (dim)
    return 1 - fid
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
