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

from .info_extract import extract_qumode_info, extract_wavefunction

# Generate and plot the time evolution of a qubit-qumode state in the position basis -----
# ----------------------------------------------------------------------------------------
def generate_wavefunc(init_qubit_state, init_qumode_state, target_time, n_steps, gen_hamiltonian, mp, N, morse_to_optimize):
    """    
    Args:
        init_qubit_state: ndarray, initial qubit state vector
        init_qumode_state: ndarray, initial qumode state vector in Fock basis
        target_time: float, final time for evolution
        n_steps: int, number of time points to evaluate
        gen_hamiltonian: ndarray, Hamiltonian for synthesized evolution
        mp: object, contains mass, angular frequency, hbar for Morse potential
        N: int, dimension of qumode Fock space
        morse_to_optimize: ndarray, target unitary for Morse potential evolution
    """

    tgt_hamiltonian = (1j / target_time) * sp.linalg.logm(morse_to_optimize)  # convert target unitary to effective Hamiltonian
    x_var = np.linspace(-2, 8, 200)                                           # positions where psi(x) will be evaluated
    init_state = np.kron(init_qubit_state, init_qumode_state)                 # Create full initial state in the joint qubit-qumode Hilbert space
    times = np.linspace(0, target_time, n_steps)                              # time points for the time evolution

    # Setup plot grid
    cols = 3
    rows = (n_steps + cols - 1) // cols                                       
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), constrained_layout=True)
    axes = axes.flatten()                                                      

    # Loop through each time step
    for i, t in enumerate(times):
        ax = axes[i]                                                          # current axis to plot on

        # --- Evolve under synthesized Hamiltonian ---
        U_gen_t = sp.linalg.expm(-1j * gen_hamiltonian * t)                   # time evolution operator
        psi_gen_t = U_gen_t @ init_state                                      # evolve full hybrid state
        psi_gen, norm_gen = extract_qumode_info(psi_gen_t, 0, N)                     # extract qumode component for qubit |0>
        psi_gen_x = extract_wavefunction(psi_gen, x_var, mp) * norm_gen                  # project qumode to position basis
        #psi_gen_x /= np.sqrt(np.sum(np.abs(psi_gen_x)**2) * (x_var[1]-x_var[0]))
        

        # --- Evolve under target Hamiltonian ---
        U_tgt_t = sp.linalg.expm(-1j * tgt_hamiltonian * t)                   # target evolution operator
        psi_tgt_t = U_tgt_t @ init_state                                      # evolve full hybrid state
        psi_tgt, norm_tgt = extract_qumode_info(psi_tgt_t, 0, N)                     # extract qumode component
        psi_tgt_x = extract_wavefunction(psi_tgt, x_var, mp) * norm_tgt               # project to position basis
        #psi_tgt_x /= np.sqrt(np.sum(np.abs(psi_tgt_x)**2) * (x_var[1]-x_var[0]))


        # --- Plot ---
        ax.plot(x_var, np.abs(psi_gen_x)**2, label='Synthesized', color='red') # plot probability density for generated evolution
        ax.plot(x_var, np.abs(psi_tgt_x)**2, label='Target', color='blue', linestyle='--') # plot target
        ax.set_title(f'Time = {round(t, 2)}')                                 # set subplot title
        ax.set_xlabel('x')                                                    # x-axis label
        ax.set_ylabel(r'$|\psi(x,t)|^2$')                                     # y-axis label
        ax.grid(True)                                                          # add grid

    # Remove unused axes if n_steps < rows*cols
    for j in range(len(times), len(axes)):
        fig.delaxes(axes[j])

    # Add shared legend and super-title
    handles, labels = axes[0].get_legend_handles_labels()                     # get handles from first subplot
    fig.legend(handles, labels, loc='upper left', ncol=2, fontsize='medium') # shared legend
    plt.suptitle('Time Evolution of State in Position Basis', fontsize=16)    # figure title
    plt.show()                                                                 # display figure
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------