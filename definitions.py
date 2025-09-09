import numpy as np
import scipy as sp
import constants
import matplotlib.pyplot as plt

"""
Definining the Operators
- Morse Hamiltonian
- Always-on Hamiltonian
- Displacement Gate
- Qubit xy Rotation Gate
"""

"""
Function which defines the constants that will be used
"""



I_q = np.eye(2) #Identity operator for qubits
I_o = np.eye(N) #Identity operator for qumodes

a = np.diag(np.sqrt(np.arange(1, N)), 1)  # Annihilation operator
adag = a.T.conj()                         # Creation operator
n_op = adag @ a                           # Photon number operator

# Pauli Matrices for qubit
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

H_always_on = (chi * sigma_z + omega * np.eye(2))  # Always-on Hamiltonian for a cavity-QED System
H_always_on = np.kron(H_always_on, n_op)           # Full Always-on Hamiltonian (Qubit-Qumode Hilbert Space)

# Always-on Time Evolution
def H_On_Evo(t):
    return sp.linalg.expm(-1j * H_always_on * t)

# Dispacement Gate 
def displacement(alpha):
    A = alpha * adag - np.conj(alpha) * a
    return sp.linalg.expm(A)

# Dispacement Gate (Qubit-Qumode Hilbert Space)
def D_full(alpha):
    return np.kron(I_q, displacement(alpha))

# Qubit XY Rotation Gate
def R_phi(theta, phi):
    op = (np.cos(phi) * sigma_x + np.sin(phi) * sigma_y)
    return sp.linalg.expm(-1j * theta / 2 * op)

# Qubit XY Rotation Gate (Qubit-Qumode Hilbert Space)
def R_full(theta, phi):
    return np.kron(R_phi(theta, phi), I_o)


# Morse Hamiltonian (Returns the Matrix from the Qubit-Qumode Hilbert Space)
def H_Morse(u ,de, b, x0):
    """
    Morse Hamiltonian Gate

    Args:
        u (real): reduced mass of diatomic system
        de (real): dissociation energy
        b (real): width parameter
        x0 (real): equilibrium bond length

    Returns:
        csc_matrix: operator matrix
    """
    m_omega = np.sqrt(2 * de * (b ** 2) / u)
    X_op = (a + adag) / np.sqrt(2)
    P_op = 1j * (a - adag) / (np.sqrt(2))
    x_op = X_op * np.sqrt(hbar / (u * m_omega))
    p_op = P_op * np.sqrt(hbar * u * omega)

    kin_op = (p_op @ p_op) / (2 * u)

    exp_term = sp.linalg.expm(-1 * b * (x_op - x0 * np.eye(N)))
    mp_op = de * ((np.eye(N) - exp_term) @ (np.eye(N) - exp_term))

    full_m = kin_op + mp_op

    full_m = np.kron(I_q, full_m)

    return full_m

# Time Evolution for the Morse Hamiltonian
def MH_Evo(t, u, de, b, x0):
    return sp.linalg.expm(-1j * H_Morse(u, de, b, x0) * t)

# Build the Gate Sequence (Always-On Time Evolution, Displacement Gate, XY Rotation Gate)
def gate_seq(params, d):
    """
    Builds the Gate Sequence

    Params: Each layer is a list of 4 parameters
    1. alpha_real - real part of alpha which shifts position
    2. alpha_imag - imaginary part of alpha which shifts momentum
    3. theta - one of the parameters for the xy qubit rotation gate
    4. phi - one of the parameters for the xy qubit rotaation gate
    
    """
    U = np.eye(2 * N, dtype=complex)
    for j in range(d):
        alpha_real = params[4*j]
        alpha_imag = params[4*j+1]
        theta = params[4*j+2]
        phi = params[4*j+3]
        
        D = D_full(alpha_real + 1j * alpha_imag)
        R = R_full(theta, phi)
        V = H_On_Evo(time)

        U = U @ V @ R @ D
    return U

# Definition of the cost (infidelity) function
def fidelity_loss(params, d, U_target):
    U = gate_seq(params, d)
    dim = U.shape[0]
    fid = np.abs(np.trace(U.conj().T @ U_target))/ (dim)
    return 1 - fid

# Create an Instance of the Target Morse Hamiltonian Time Evolution
morse_to_optimize = MH_Evo(t=time, u=mass, de=diss_energy, b=width_param, x0=equib_length)

#Plot the Morse Potential

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