"""
Filename: morse.py
Author: Alexay Mehra
Date: 2025-09-28
Description: Class to hold constants and parameters for the Morse Potential.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt


# Morse Potential Class ------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
class MorsePotential:


    def __init__(self, de=8.0, b=1.0, x0=0.0):
        """
        Define the paramaters for the Morse Potential
        Args:
            de: well depth
            b: width parameter
            x0:equilibrium bond length
        """
        self.de = de
        self.b = b
        self.x0 = x0


    def plot_potential(self):
        """
        Plot the Morse Potential based on the defined parameters
        """
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