import numpy as np

# Time to optimize over
time = 1

# Parameters for Morse Potential
mass = 1
diss_energy = 8
width_param = 1
equib_length = 0

# Oscillator Parameters
omega = 1.0  # Oscillator frequency
chi = 0.01   # Qubit-oscillator coupling

# Other Parameters
hbar = 1 

# Trruncate the Fock Space Based on the number of states supported by the Morse Potential
morse_cap = (int) (np.sqrt(2 * mass * diss_energy) / (width_param * hbar) - (1/2))


N = 10