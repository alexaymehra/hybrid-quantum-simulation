"""
Filename: optimize_info.py
Author: Alexay Mehra
Date: 2025-10-07
Description: Contains the funciton to build the gate sequence and the cost function for optimization.
"""


# Imports
import numpy as np
import qutip


# Building a generic gate sequence -------------------------------------------------------
# ----------------------------------------------------------------------------------------
def build_sequence(params, seq_template, backend, d=1):
    """
    Builds the full gate to optimize sequence for d layers:
    Args:
        params: flat array of optimization parameters
        seq_template: list of GateSpec objects representing a single layer
        backend: backend object with operators
        d: number of times to repeat the layer
    Returns:
        U: full sequence of gates to optimize
    """
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
def infidelity(params, seq_template, backend, d, target_evolution, init_state, trace_index):
    """
    Infidelity between synthesized evolution and true evolution
    Args:
        params: flat array of optimization parameters
        seq_template: list of GateSpec objects representing a single layer
        backend: backend object with operators
        d: number of times to repeat the layer
        target_evolution: `ndarray` evolution that is trying to be matched
        init_state: 'Qobj' initial qubit-qumode state to evolve
        trace_index: an `int` to indicate what component to keep from partial trace
    Returns:
        infid: number between 0 and 1 representing the infidelity beteen the synthesized and true evolution on an initial state
    """
    synth_evolution = build_sequence(params, seq_template, backend, d)
    
    # evolve the initial state
    synth_state = synth_evolution @ (init_state.full())
    true_state = target_evolution @ (init_state.full())

    # extract just the qumode information from the state through ptrace
    synth_qobj = qutip.Qobj(synth_state, dims=[[2, backend.dim], [1, 1]])
    real_qobj = qutip.Qobj(true_state, dims=[[2, backend.dim], [1, 1]])
    synth_qumode = synth_qobj.ptrace(trace_index)
    true_qumode = real_qobj.ptrace(trace_index)

    fid = qutip.fidelity(synth_qumode, true_qumode)

    infid = 1 - fid
  
    return infid
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
