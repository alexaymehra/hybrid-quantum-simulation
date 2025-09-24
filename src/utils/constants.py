"""
Filename: constants.py
Author: Alexay Mehra
Date: 2025-09-10
Description: Holds constants to be used in operators.
"""

# Imports
import numpy as np


class MorsePotential():
    """
    Create an Instance of the Morse Potential:
    m = mass
    de = well depth
    b = width parameter
    x0 = equilibrium length of the bond
    hbar = Reduced Planck's Constant (default 1 for easy units)
    """
    def __init__(self, mass=1, de=8, b=1, x0=0, hbar=1):
        self.mass = mass
        self.de = de
        self.b = b
        self.x0 = x0
        self.hbar = hbar
        self.morse_cap = (int) (np.sqrt(2 * mass * de) / (b * hbar) - (1/2))
        return

time = 1 # Time to optimize over

# Oscillator Parameters
omega = 1.0  # Oscillator frequency
chi = 0.01   # Qubit-oscillator coupling

N = 10