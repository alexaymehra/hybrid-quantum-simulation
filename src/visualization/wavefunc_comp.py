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

from .info_extract import extract_qumode_info, fock_basis_to_position


def generate_wavefunc(init_state, target_time, n_steps, gen_hamiltonian, backend, morse_to_optimize, mp):

    # Extract the Hamiltonian from the Time Evolution
    tgt_hamiltonian = (1j / backend.time) * sp.linalg.logm(morse_to_optimize)

    # Parameters
    x_var = np.linspace(-2, 8, 200)
    topos = fock_basis_to_position(x_var, backend.dim, mp)

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
        psi_gen = extract_qumode_info(psi_gen_t, 0, backend.dim)
        psi_gen_x = topos @ psi_gen

        # Evolve under true
        U_tgt_t = sp.linalg.expm(-1j * tgt_hamiltonian * t)
        psi_tgt_t = U_tgt_t @ init_state
        psi_tgt = extract_qumode_info(psi_tgt_t, 0, backend.dim)
        psi_tgt_x = topos @ psi_tgt

        # Normalize on the grid
        dx = x_var[1] - x_var[0]
        psi_gen_x /= np.sqrt(np.sum(np.abs(psi_gen_x)**2) * dx)
        psi_tgt_x /= np.sqrt(np.sum(np.abs(psi_tgt_x)**2) * dx)

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