"""
Filename: morse.py
Author: Alexay Mehra
Date: 2025-09-28
Description: Class to hold constants and parameters for the morse potential
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt


# Morse Potential Class ------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
class MorsePotential:
    """
    Create an Instance of the Morse Potential:
    mass = reduced mass of diatomic system
    de = well depth
    b = width parameter
    x0 = equilibrium bond length
    hbar = Reduced Planck's Constant (default 1 for easy units)
    """
    def __init__(self, mass=1.0, de=8.0, b=1.0, x0=0.0, hbar=1.0):
        self.mass = mass
        self.de = de
        self.b = b
        self.x0 = x0
        self.hbar = hbar
        self.force_const = 2 * de * (b ** 2)
        self.angular_freq = b * np.sqrt(2 * de / mass)
        lamda = np.sqrt(2 * mass * de) / (b * hbar)
        self.num_bound_states = np.floor(lamda - (0.5)) + 1
        self.max_q_number = self.num_bound_states - 1

    def plot_potential(self):
        x = np.linspace(-2, 8, 200)
        target_potential = self.de * ((1 - np.exp(-1 * self.b * (x - self.x0))) ** 2)
        plt.figure(figsize=(8, 4))
        plt.plot(x, target_potential, label='Target Morse Potential')
        plt.xlabel('x')
        plt.ylabel('V(x)')
        plt.ylim(0, self.de * 1.3)
        plt.title('Morse Potential')
        plt.grid(True)
        plt.legend()
        plt.show()
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------