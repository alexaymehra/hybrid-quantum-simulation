"""
Filename: plotpot.py
Author: Alexay Mehra
Date: 2025-09-08
Description: Contains function which generates the graph of the morse potential based on the given parameters
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

from constants import diss_energy, width_param, equib_length


def plot_potential():
    # Position axis
    x = np.linspace(-2, 8, 200)
    # Morse potential
    V_Target = diss_energy * (1 - np.exp(-width_param * (x - equib_length)))**2
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(x, V_Target, label='Target Morse Potential')
    plt.xlabel('x')
    plt.ylabel('V(x)')
    plt.ylim(0, diss_energy * 1.3)  # Adjust the 1.2 factor as needed
    plt.title('Morse Potential')
    plt.grid(True)
    plt.legend()
    plt.show()
    return
