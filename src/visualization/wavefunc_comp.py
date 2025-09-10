"""
Filename: optimize_func.py
Author: Alexay Mehra
Date: 2025-09-10
Description: Contains functions to compare generated vs target wavefunction evolution
"""

# Imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from utils.constants import mass, hbar, omega, time, N
from optimization.optimize_info import morse_to_optimize


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
def fock_basis_to_position(x, N, m=mass, hbar=hbar, omega=omega):
    """
    Returns T[i, n] = ⟨x_i | n⟩, i.e., harmonic oscillator wavefunction for n-th Fock state at position x_i
    """
    xi = np.sqrt(m * omega / hbar) * x
    prefactor = (m * omega / (np.pi * hbar))**0.25
    T = np.zeros((len(x), N), dtype=complex)

    for n in range(N):
        norm = 1 / np.sqrt(2**n * sp.special.factorial(n))
        Hn = sp.special.eval_hermite(n, xi)
        psi_n = prefactor * norm * np.exp(-0.5 * xi**2) * Hn
        T[:, n] = psi_n

    return T

def generate_wavefunc(init_qubit_state, init_qumode_state, target_time, n_steps, gen_hamiltonian):

    # Extract the Hamiltonian from the Time Evolution
    tgt_hamiltonian = (1j / time) * sp.linalg.logm(morse_to_optimize)

    # Parameters
    x_var = np.linspace(-2, 8, 200)
    topos = fock_basis_to_position(x_var, N)

    # Create the Initial State in the Full Qubit-Qumode Hilbert Space
    init_state = np.kron(init_qubit_state, init_qumode_state)

    # --- Time Steps ---
    times = np.linspace(0, target_time, n_steps)

    # --- Setup plot grid ---
    cols = 3
    rows = (n_steps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), constrained_layout=True)
    axes = axes.flatten()

    # --- Loop through time evolution ---
    for i, t in enumerate(times):
        ax = axes[i]

        # Evolve under synthesized
        U_gen_t = sp.linalg.expm(-1j * gen_hamiltonian * t)
        psi_gen_t = U_gen_t @ init_state
        psi_gen = extract_qumode_info(psi_gen_t, 0, N)
        psi_gen_x = topos @ psi_gen

        # Evolve under true
        U_tgt_t = sp.linalg.expm(-1j * tgt_hamiltonian * t)
        psi_tgt_t = U_tgt_t @ init_state
        psi_tgt = extract_qumode_info(psi_tgt_t, 0, N)
        psi_tgt_x = topos @ psi_tgt

        # Plot
        ax.plot(x_var, np.abs(psi_gen_x)**2, label='Synthesized', color='red')
        ax.plot(x_var, np.abs(psi_tgt_x)**2, label='Target', color='blue', linestyle='--')
        ax.set_title(f'Time = {round(t, 2)}')
        ax.set_xlabel('x')
        ax.set_ylabel(r'$|\psi(x,t)|^2$')
        ax.grid(True)

    # Remove unused axes
    for j in range(len(times), len(axes)):
        fig.delaxes(axes[j])

    # Shared legend and title
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', ncol=2, fontsize='medium')
    plt.suptitle(f'Time Evolution of State in Position Basis', fontsize=16)
    plt.show()
    return