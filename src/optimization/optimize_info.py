"""
Filename: gate_specs.py
Author: Alexay Mehra
Date: 2025-09-28
Description: 
"""


# Imports
import numpy as np

from visualization.info_extract import extract_qumode_info, fock_basis_to_position

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
def fidelity_loss(params, seq_template, backend, d, U_target, info_type, init_state, qubit_state_index, mp):
    # build the sequence based on gate parameters
    U = build_sequence(params, seq_template, backend, d)
    
    # evolve the initial state
    synth_state = U @ init_state
    true_state = U_target @ init_state

    # extract qumode information from evolved state & normalize
    synth_qumode = extract_qumode_info(synth_state, qubit_state_index, backend.dim)
    true_qumode = extract_qumode_info(true_state, qubit_state_index, backend.dim)
    synth_qumode /= np.linalg.norm(synth_qumode)
    true_qumode /= np.linalg.norm (true_qumode)

    # optionally only look at position information
    if (info_type == 'position'):
        nx = 400
        xmax = 10
        xvec = np.linspace(-1 * xmax, xmax, nx)
        pos_proj = fock_basis_to_position(xvec, backend.dim, mp)
        synth_qumode = pos_proj @ synth_qumode
        true_qumode = pos_proj @ true_qumode
        
        dx = xvec[1] - xvec[0]
        
        synth_qumode /= np.sqrt(np.sum(np.abs(synth_qumode)**2) * dx)
        true_qumode /= np.sqrt(np.sum(np.abs(true_qumode)**2) * dx)

    fid = np.abs(np.vdot(synth_qumode, true_qumode))  # vdot handles conjugation automatically

    return 1 - fid
# --------------------------------------s--------------------------------------------------
# ----------------------------------------------------------------------------------------
