"""
Filename: simulate.py
Author: Alexay Mehra
Date: 2025-10-07
Description: Contains the functions to setup the main simulations.
"""

import qutip
import numpy as np


def simulate(init_state, full_hamiltonian=None, end_time=None, backend=None, evolution=None, times=None):
    """
    Simulates the evolution of an initial state based on a given Hamiltonian or evolution operator
    Args:
        init_state: `Qobj` full initial state in qubit-qumode space
        full_hamiltonian: `ndarray` Hamiltonian used to evolve state in qubit-qumode space (Mode A)
        end_time: the time to evolve the state until (Mode A)
        backend: backend for the simulation (Mode A)
        evolution: `ndarray` or list of evolution operators (Mode B)
        times: custom time array (Mode B) or None for default (Mode A)
    Returns:
        result: `dict` with evolved qubit-qumode states at different times ("states"), and the different times ("times")
    Note:
        This function assumes the ordering of the joint state is (qubit, qumode)
        Mode A: Uses full_hamiltonian to compute evolution at multiple times
        Mode B: Uses pre-computed evolution operator(s)
   """
    
    if full_hamiltonian is not None:
        # Mode A: Hamiltonian-based simulation (original behavior)
        if end_time is None:
            raise ValueError("end_time must be provided when using full_hamiltonian")
        if backend is None:
            raise ValueError("backend must be provided when using full_hamiltonian")
            
        if times is None:
            times = np.linspace(0.0, end_time, 50)
        
        states = []
        unitaries = []

        for n in range(0, len(times)):
            hamiltonian_qobj = qutip.Qobj(full_hamiltonian, dims=[[2, backend.dim],[2, backend.dim]])
            evolution_at_time = (-1j * hamiltonian_qobj * times[n]).expm()
            state_at_time = evolution_at_time * init_state
            states.append(state_at_time)
            unitaries.append(evolution_at_time)

        result = {"states": states, "times": times}
        
    elif evolution is not None:
        # Mode B: Pre-computed evolution operator(s)
        if isinstance(evolution, list):
            # Multiple evolution operators with corresponding times
            if times is None:
                raise ValueError("times must be provided when evolution is a list")
            if len(evolution) != len(times):
                raise ValueError("evolution and times lists must have the same length")
            
            states = []
            for i, evo in enumerate(evolution):
                state_at_time = evo * init_state
                states.append(state_at_time)
        else:
            # Single evolution operator - apply once
            if times is None:
                times = [1.0]  # Default single time
            elif isinstance(times, (int, float)):
                times = [times]  # Convert single time to list
            elif len(times) != 1:
                raise ValueError("Single evolution operator requires single time or None")
            
            states = [evolution * init_state]
        
        result = {"states": states, "times": times}
    else:
        raise ValueError("Either full_hamiltonian or evolution must be provided")
    
    return result
    