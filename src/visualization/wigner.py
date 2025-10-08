"""
Filename: optimize_func.py
Author: Alexay Mehra
Date: 2025-09-10
Description: Contains a funciton to plot the wigner function for different states in time.
"""


# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import qutip


def plot_wigner_fucntion(result, trace_index):
    """
    Plots the Wigner Function of a Qubit-Qumode State Vector
    Args:
        result: a `dict` with indices "states" and "times" where "states" is the qubit-qumode state at the different times indicated by "times" 
        trace_index: an `int` to indicate what component to keep from partial trace
    """

    # --- 1. Setup indicies and grid ---
    num_plots = 12
    indices = np.linspace(0, len(result["states"]) - 1, num_plots, dtype=int)
    lim_axis = 9
    xvec = np.linspace(-lim_axis, lim_axis, 100)

    # --- 2. Computer Wigner functions ---
    wigner_data = []
    for i in indices:
        state = result["states"][i]
        # ptrace used to only keep subsystem you want (ex: qumode)
        # ptrace returns a density matrix for the state
        qumode_state = state.ptrace(trace_index)
        wig_var = qutip.wigner(qumode_state, xvec, xvec)   
        wigner_data.append(wig_var)

    # --- 3. Normalize color scale across all plots ---
    W_all = np.array(wigner_data)
    vmax = np.max(np.abs(W_all))
    norm = colors.Normalize(vmin=-vmax, vmax=vmax)

    # --- 4. Create subplot grid ---
    fig, axes = plt.subplots(2, 6, figsize=(14, 4))
    cmap = 'seismic'
    times = np.array(result["times"])

    # --- 5. Fill each subplot ---
    for idx, (ax, W, i) in enumerate(zip(axes.flat, wigner_data, indices)):
        cf = ax.contourf(xvec, xvec, W, 100, norm=norm, cmap=cmap)
        ax.set_title(r't = {:.2}'.format(times[indices][idx]))

    # --- 6. Clean up axis ticks/labels ---
    if idx < 6:
        ax.set_xticks([])
        ax.set_xlabel('')
    if idx % 6 != 0:
        ax.set_yticks([])
        ax.set_ylabel('')

    # --- 7. Add shared colorbar ---
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    fig.colorbar(cf, cax=cbar_ax)

    # --- 8. Adjust layout & show ---
    plt.subplots_adjust(wspace=0.02, hspace=0.01, right=0.9)
    plt.show()