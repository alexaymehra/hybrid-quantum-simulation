"""
Filename: constants.py
Author: Alexay Mehra
Date: 2025-09-08
Description: Holds constants to be used in operators.
"""

# Imports
import numpy as np

time = 1 # Time to optimize over

# Parameters for Morse Potential
mass = 1
diss_energy = 8
width_param = 1
equib_length = 0

# Oscillator Parameters
omega = 1.0  # Oscillator frequency
chi = 0.01   # Qubit-oscillator coupling

# Parameters for units
hbar = 1 

# Trruncate the Fock Space Based on the number of states supported by the Morse Potential
morse_cap = (int) (np.sqrt(2 * mass * diss_energy) / (width_param * hbar) - (1/2))


N = 10