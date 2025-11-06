"""
Filename: optimize_info.py
Author: Alexay Mehra
Date: 2025-10-14
Description: Contains the funciton to build the gate sequence and the cost function for optimization.
"""


# Imports
import numpy as np
import scipy as sp
import qutip


# Building a generic gate sequence -------------------------------------------------------
# ----------------------------------------------------------------------------------------
def build_sequence(params, seq_template, backend, d=1):
    """
    Builds the full gate sequence to optimize sequence for d layers:
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


# Building a sequence for a specific time -------------------------------------------------
# ----------------------------------------------------------------------------------------
def build_sequence_for_time(params, seq_template, backend, d, t):
    """
    Builds the full gate sequence for a specific time t, incorporating time evolution
    Args:
        params: flat array of optimization parameters
        seq_template: list of GateSpec objects representing a single layer
        backend: backend object with operators
        d: number of times to repeat the layer
        t: time to evolve for
    Returns:
        U: full sequence of gates evolved for time t
    """
    tot_dim = backend.I_o.shape[0] * backend.I_q.shape[0]   # total hilbert space dimension
    U = np.eye(tot_dim, dtype=complex)                      # starting point for the sequence
    idx = 0                                                 # keeps track of position in params

    full_seq = seq_template * d                             # repeat the sequence d times

    for gate in full_seq:                                   # loop through all gates in the sequence
        gate_params = params[idx: idx + gate.n_params]      # slice out parameters for the current gate
        
        if gate.name == "Controlled Rotation":
            gate.config["time"] = t / d

        gate_matrix = gate.build(gate_params, backend)      # build the gate
        
        if gate.name == "Controlled Rotation":
            gate_matrix = backend.c_r(omega = 0, chi = 2, t = t / d)
        
        U = U @ gate_matrix                                 # multiply the gate into the sequence
        idx += gate.n_params                                # increment idx

    return U                                                # return full sequence
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------


# Cost (Infidelity) Function -------------------------------------------------------------
# ----------------------------------------------------------------------------------------
def infidelity(params, seq_template, backend, d, target_evolution, init_state, trace_index, times=None):
    """
    Infidelity between synthesized evolution and true evolution
    Args:
        params: flat array of optimization parameters
        seq_template: list of GateSpec objects representing a single layer
        backend: backend object with operators
        d: number of times to repeat the layer
        target_evolution: `ndarray` or list of evolutions that are trying to be matched
        init_state: 'Qobj' initial qubit-qumode state to evolve
        trace_index: an `int` to indicate what component to keep from partial trace
        times: optional array of times for multi-time optimization (default None for single-time)
    Returns:
        infid: number between 0 and 1 representing the infidelity between the synthesized and true evolution on an initial state
    """
    
    if times is None:
        # Single-time mode (original behavior)
        synth_evolution = build_sequence(params, seq_template, backend, d)

        synth_state = synth_evolution @ (init_state.full())
        
        # Ensure target_evolution is a numpy array for matrix multiplication
        if hasattr(target_evolution, 'full'):
            # It's a Qobj, convert to numpy array
            true_state = target_evolution.full() @ (init_state.full())
        else:
            # It's already a numpy array
            true_state = target_evolution @ (init_state.full())

        # extract just the qumode information from the state through ptrace
        synth_qobj = qutip.Qobj(synth_state, dims=[[2, backend.dim], [1, 1]])
        real_qobj = qutip.Qobj(true_state, dims=[[2, backend.dim], [1, 1]])
        synth_qumode = synth_qobj.ptrace(trace_index)
        true_qumode = real_qobj.ptrace(trace_index)

        fid = qutip.fidelity(synth_qumode, true_qumode)
        infid = 1 - fid

    else:
        # Multi-time mode (time-averaged fidelity)
        my_fidelity = []
        
        # Check if target_evolution is a list of evolutions for each time
        if isinstance(target_evolution, list):
            if len(target_evolution) != len(times):
                raise ValueError("target_evolution list must have same length as times")
            target_evolutions = target_evolution
        else:
            # Single target evolution - scale it for each time (fallback)
            target_evolutions = [sp.linalg.expm(sp.linalg.logm(target_evolution) * t) for t in times]
        
        for i, t in enumerate(times):
            # Build sequence for this specific time
            synth_evolution_t = build_sequence_for_time(params, seq_template, backend, d, t)
            
            # Use the corresponding target evolution for this time
            target_evolution_t = target_evolutions[i]
            
            # evolve the initial state
            synth_state = synth_evolution_t @ (init_state.full())
            true_state = target_evolution_t @ (init_state.full())

            # extract just the qumode information from the state through ptrace
            synth_qobj = qutip.Qobj(synth_state, dims=[[2, backend.dim], [1, 1]])
            real_qobj = qutip.Qobj(true_state, dims=[[2, backend.dim], [1, 1]])
            synth_qumode = synth_qobj.ptrace(trace_index)
            true_qumode = real_qobj.ptrace(trace_index)

            fid = qutip.fidelity(synth_qumode, true_qumode)
            my_fidelity.append(fid)
        
        # Average fidelity over all times
        avg_fid = np.sum(my_fidelity) / len(times)
        infid = 1 - avg_fid
    
    print(f"Current infidelity: {infid}")
    return infid
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

def new_cost(params, seq_template, backend, d, target_evolution, init_state, trace_index, times):
    """Temporary cost function for testing.
    """

    overall_fidelity = []

    for i in range(0, len(times)):
        synth_evolution = build_sequence_for_time(params, seq_template, backend, d, times[i])
        synth_state = synth_evolution @ (init_state.full())
        true_state = target_evolution[i] @ (init_state.full())

        # extract just the qumode information from the state through ptrace
        synth_qobj = qutip.Qobj(synth_state, dims=[[2, backend.dim], [1, 1]])
        real_qobj = qutip.Qobj(true_state, dims=[[2, backend.dim], [1, 1]])
        synth_qumode = synth_qobj.ptrace(trace_index)
        true_qumode = real_qobj.ptrace(trace_index)
        fid = qutip.fidelity(synth_qumode, true_qumode)
        overall_fidelity.append(fid)
    
    avg_fid = np.sum(overall_fidelity) / len(times)
    infid = 1 - avg_fid
    infid = 100 * infid
    print(f"Current infidelity: {infid}")
    return infid

def optimize_new_cost(seq_template, backend, d, target_evolution, init_state, trace_index, times):
    """Temporary optimization function for testing.
    """

    # total number of parameters across d layers
    n_params_per_layer = sum(g.n_params for g in seq_template)
    total_params = n_params_per_layer * d

    #init_guess = (np.random.rand(total_params) - 0.5) *   # range [-1, 1]
    init_guess = np.zeros(total_params)

    result = sp.optimize.minimize(
        new_cost,          
        init_guess,
        args=(seq_template, backend, d, target_evolution, init_state, trace_index, times),
        method='COBYLA',
        options={'disp': True, 'maxiter': 1000}
    )

    return result.x