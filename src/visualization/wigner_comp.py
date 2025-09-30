"""
Filename: optimize_func.py
Author: Alexay Mehra
Date: 2025-09-10
Description: Contains functions to compare generated vs target wigner function evolution
"""

# Imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from qutip import Qobj, wigner

from .info_extract import extract_qumode_info

def gen_wigfunc(init_state, time_and_extra, steps, generated_hamiltonian, backend, morse_to_optimize):
        

    # Extract the Hamiltonian from the Time Evolution
    target_hamiltonian = (1j / backend.time) * sp.linalg.logm(morse_to_optimize)


    time_step = time_and_extra / steps

    # Define phase space grid
    xvec = np.linspace(-5, 5, 200)

    fig, axes = plt.subplots(2, steps+1, figsize=(3 * (steps+1), 6), constrained_layout=True)
    fig.suptitle('Wigner Function Evolution: Top = Target, Bottom = Generated', fontsize=14)

    for i in range(steps+1):
        t = i * time_step
        U_gen_t_wig = sp.linalg.expm(-1j * generated_hamiltonian * t)
        U_tgt_t_wig = sp.linalg.expm(-1j * target_hamiltonian * t)

        step_state_generated = U_gen_t_wig @ init_state
        step_state_target = U_tgt_t_wig @ init_state

        # Calculate Fidelity at Each Step
        fidelity = np.abs(np.vdot(step_state_generated, step_state_target))**2
        print(f"Fidelity at t = {t:.2f}: {fidelity:.6f}")

        qumode_generated = extract_qumode_info(step_state_generated, qubit_state_index=0, qumode_dim=backend.dim)
        qumode_target = extract_qumode_info(step_state_target, qubit_state_index=0, qumode_dim=backend.dim)

        # Convert to Qobj
        step_qobj_generated = Qobj(qumode_generated, dims=[backend.dim, [1]])
        step_qobj_target = Qobj(qumode_target, dims=[backend.dim, [1]])

        # Compute Wigner functions
        W_gen = wigner(step_qobj_generated, xvec, xvec)
        W_tar = wigner(step_qobj_target, xvec, xvec)

        # Top row: target
        ax_tar = axes[0, i]
        im1 = ax_tar.contourf(xvec, xvec, W_tar, 100, cmap='RdBu_r')
        ax_tar.set_title(f'Target: t={round(i*time_step, 2)}')
        ax_tar.set_xticks([])
        ax_tar.set_yticks([])

        # Bottom row: generated
        ax_gen = axes[1, i]
        im2 = ax_gen.contourf(xvec, xvec, W_gen, 100, cmap='RdBu_r')
        ax_gen.set_title(f'Generated: t={round(i*time_step, 2)}')
        ax_gen.set_xticks([])
        ax_gen.set_yticks([])

    # Add single shared colorbar using a separate figure method
    fig.colorbar(im1, ax=axes, location='right', shrink=0.8, label='Wigner Function Value')

    plt.show()