"""
Filename: simulate.py
Author: Alexay Mehra
Date: 2025-10-07
Description: Contains the functions to setup the main simulations.
"""

import qutip
import numpy as np


def simulate(init_state, full_hamiltonian, end_time, backend):
    """
    Simulates the evolution of an initial state based on a given Hamiltonian
    Args:
        init_state: `Qobj` full initial state in qubit-qumode space
        full_hamiltonian: `ndarray` Hamiltonian used to evolve state in qubit-qumode space
        end_time: the time to evolve the state until
        backend: backend for the simulation
    Returns:
        result: `dict` with evolved qubit-qumode states at different times ("states"), and the different times ("times")
    Note:
        This function assumes the ordering of the joint state is (qubit, qumode)
   """

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
    
    return result
    