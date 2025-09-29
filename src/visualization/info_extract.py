"""
Filename: info_extract.py
Author: Alexay Mehra
Date: 2025-09-10
Description: Contains functions to compare generated vs target wavefunction evolution
"""


# Imports
import numpy as np
import scipy as sp


# Extract the Qumode Info from the full Qubit-Qumode State -------------------------------
# ----------------------------------------------------------------------------------------
def extract_qumode_info(hybrid_state, qubit_state_index, qumode_dim, eps=1e-12):
    """
    Args:
        hybrid_state: full state vector of shape (2 * qumode_dim,)
        qubit_state_index: 0 or 1
        qumode_dim: dimension of the qumode Fock space

    Returns:
        qumode_state: complex ndarray of shape (qumode_dim,)
    """

    # error catching
    assert qubit_state_index in (0, 1), "Qubit index must be 0 or 1."
    assert hybrid_state.shape[0] == 2 * qumode_dim, "Hybrid state has wrong shape."

    # extract qumode information from upper left or lower right block
    start = qubit_state_index * qumode_dim
    end = start + qumode_dim
    qumode_state = hybrid_state[start:end]

    # compute the norm of qumode state
    norm = np.linalg.norm(qumode_state)

    if norm < eps:
        # Return zero vector if probability is effectively zero
        return np.zeros_like(qumode_state), 0.0
    
    # return the normalized qumode state and the norm used to normalize
    return qumode_state / norm, norm
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------


# Project Qumode Info to the Position Basis ----------------------------------------------
# ----------------------------------------------------------------------------------------
def extract_wavefunction(qumode_state, positions, mp):
    """
    Args:
        qumode_state: ndarray of shape (dim,)
            Fock-basis coefficients of the qumode.
        positions: ndarray
            Positions to evaluate the wavefunction at.
        mp:
            Morse Potential to pass in for mass, hbar, and its angular frequency

    Returns:
        psi_x: ndarray of shape (len(x),)
            Normalized position-space wavefunction.
    """
    qumode_size = len(qumode_state)                                         # get dimension of qumode Hilbert space
    xi = np.sqrt(mp.mass * mp.angular_freq / mp.hbar) * positions           # convert position to dimensionless units
    prefactor = (mp.mass * mp.angular_freq / (np.pi * mp.hbar)) ** (1/4)    # compute global normalization factor

    T = np.zeros((len(positions), qumode_size), dtype=complex)              # initalize matrix to store position values

    for n in range(qumode_size):                                            # loop over each fock state
        norm = 1 / np.sqrt(2**n * sp.special.factorial(n))                  # compute hermite-polynomial normalization factor
        Hn = sp.special.eval_hermite(n, xi)                                 # evaluate the n-th Hermite polynomial at the dimensionless points xi
        T[:, n] = prefactor * norm * np.exp(-0.5 * xi**2) * Hn              # compute full position-space wavefunction for fock state n

    psi_x = T @ qumode_state                                                # convert fock basis vector to position-space wavefunction

    psi_x /= np.linalg.norm(psi_x)                                          # ensure the position-space function is normalized

    return psi_x
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------