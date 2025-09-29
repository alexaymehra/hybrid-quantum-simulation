"""
Filename: info_extract.py
Author: Alexay Mehra
Date: 2025-09-10
Description: Contains functions to compare generated vs target wavefunction evolution
"""


# Imports
import numpy as np
import scipy as sp


# Function to Extract the Qumode Information from the Joint Qubit-Qumode State
def extract_qumode_info(hybrid_state, qubit_state_index, qumode_dim):
    """
    Args:
        hybrid_state: full state vector of shape (2 * qumode_dim,)
        qubit_state_index: 0 or 1
        qumode_dim: dimension of the qumode Fock space

    Returns:
        qumode_state: complex ndarray of shape (qumode_dim,)
    """
    assert hybrid_state.shape[0] == 2 * qumode_dim
    start = qubit_state_index * qumode_dim
    end = start + qumode_dim
    return hybrid_state[start:end]


# Function to Project to the Position Basis
def fock_basis_to_position(x, N, mp):
    """
    Returns T[i, n] = ⟨x_i | n⟩, i.e., harmonic oscillator wavefunction for n-th Fock state at position x_i
    """
    xi = np.sqrt(mp.mass * mp.angular_freq / mp.hbar) * x
    prefactor = (mp.mass * mp.angular_freq / (np.pi * mp.hbar))**0.25
    T = np.zeros((len(x), N), dtype=complex)

    for n in range(N):
        norm = 1 / np.sqrt(2**n * sp.special.factorial(n))
        Hn = sp.special.eval_hermite(n, xi)
        psi_n = prefactor * norm * np.exp(-0.5 * xi**2) * Hn
        T[:, n] = psi_n

    return T