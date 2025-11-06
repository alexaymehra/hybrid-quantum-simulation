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


def plot_wigner_fucntion(result, trace_index, num_plots=12):
    """
    Plots the Wigner Function of a Qubit-Qumode State Vector
    Args:
        result: a `dict` with indices "states" and "times" where "states" is the qubit-qumode state at the different times indicated by "times" 
        trace_index: an `int` to indicate what component to keep from partial trace
        num_plots: number of plots to display (default=12)
    """

    # --- 1. Setup indicies and grid ---
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

    # --- 4. Create subplot grid dynamically ---
    if num_plots == 1:
        # Special case for single plot - smaller and square
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        axes = np.array([[ax]])
        rows, cols = 1, 1
    else:
        # Calculate optimal grid dimensions - prefer wider layouts
        # Start with a reasonable number of rows and calculate cols
        if num_plots <= 4:
            rows, cols = 2, 2
        elif num_plots <= 6:
            rows, cols = 2, 3
        elif num_plots <= 9:
            rows, cols = 3, 3
        elif num_plots <= 12:
            rows, cols = 3, 4
        elif num_plots <= 16:
            rows, cols = 4, 4
        else:
            # For larger numbers, use a more systematic approach
            rows = int(np.ceil(np.sqrt(num_plots)))
            cols = int(np.ceil(num_plots / rows))
        
        # Calculate figure size to keep individual plots square
        plot_size = 2.5  # Size of each individual subplot
        fig_width = cols * plot_size
        fig_height = rows * plot_size
        
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        
        # Ensure axes is always a 2D array for consistent indexing
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
    
    cmap = 'seismic'
    times = np.array(result["times"])

    # --- 5. Fill each subplot ---
    for idx, (ax, W, i) in enumerate(zip(axes.flat, wigner_data, indices)):
        cf = ax.contourf(xvec, xvec, W, 100, norm=norm, cmap=cmap)
        ax.set_title(r't = {:.2f}'.format(float(times[indices][idx])))

        # --- 6. Clean up axis ticks/labels (fixed indentation) ---
        # For single plots, keep all axis labels
        if num_plots > 1:
            if idx < cols:  # Top row
                ax.set_xticks([])
                ax.set_xlabel('')
            if idx % cols != 0:  # Not leftmost column
                ax.set_yticks([])
                ax.set_ylabel('')
    
    # --- 6.5. Hide unused subplots ---
    total_subplots = rows * cols
    if total_subplots > num_plots:
        # Hide the unused subplots
        for idx in range(num_plots, total_subplots):
            axes.flat[idx].set_visible(False)

    # --- 7. Add shared colorbar ---
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    fig.colorbar(cf, cax=cbar_ax)

    # --- 8. Adjust layout & show ---
    plt.subplots_adjust(wspace=0.02, hspace=0.01, right=0.9)
    plt.show()