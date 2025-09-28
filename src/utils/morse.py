"""
Filename: constants.py
Author: Alexay Mehra
Date: 2025-09-28
Description: Class to hold constants and parameters for the morse potential
"""

# Imports
import numpy as np


class MorsePotential():
    """
    Create an Instance of the Morse Potential:
    mass = reduced mass of diatomic system
    de = well depth
    b = width parameter
    x0 = equilibrium bond length
    hbar = Reduced Planck's Constant (default 1 for easy units)
    omega = angular frequency of morse potential
    """
    def __init__(self, mass=1, de=8, b=1, x0=0, hbar=1, omega=1):
        self.mass = mass
        self.de = de
        self.b = b
        self.x0 = x0
        self.force_const = 2 * de * (b ** 2)
        self.angular_freq = b * np.sqrt(2 * de / mass)
        lamda = np.sqrt(2 * mass * de) / (b * hbar)
        self.num_bound_states = np.floor(lamda - (1/2)) + 1
        self.max_q_number = self.num_bound_states - 1