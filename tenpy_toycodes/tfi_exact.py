"""Provides exact ground state (and excitation) energies for the transverse field ising model.

The Hamiltonian reads
.. math ::
    H = - J \\sum_{i} \\sigma^z_i \\sigma^z_{i+1} - g \\sum_{i} \\sigma^x_i

For the exact analytical solution (in the thermodynamic limit) we use Subir Sachdev, Quantum Phase 
Transitions, 2nd ed, Cambridge University Press, 2011.
"""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
import warnings
import scipy.integrate


def finite_gs_energy(L, J, g, return_psi=False):
    """For comparison: obtain ground state energy from exact diagonalization.

    Exponentially expensive in L, only works for small enough `L` <~ 20.
    """
    if L >= 20:
        warnings.warn("Large L: Exact diagonalization might take a long time!")
    # get single site operaors
    sx = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
    sz = sparse.csr_matrix(np.array([[1., 0.], [0., -1.]]))
    id = sparse.csr_matrix(np.eye(2))
    sx_list = []  # sx_list[i] = kron([id, id, ..., id, sx, id, .... id])
    sz_list = []
    for i_site in range(L):
        x_ops = [id] * L
        z_ops = [id] * L
        x_ops[i_site] = sx
        z_ops[i_site] = sz
        X = x_ops[0]
        Z = z_ops[0]
        for j in range(1, L):
            X = sparse.kron(X, x_ops[j], 'csr')
            Z = sparse.kron(Z, z_ops[j], 'csr')
        sx_list.append(X)
        sz_list.append(Z)
    H_zz = sparse.csr_matrix((2**L, 2**L))
    H_x = sparse.csr_matrix((2**L, 2**L))
    for i in range(L - 1):
        H_zz = H_zz + sz_list[i] * sz_list[(i + 1) % L]
    for i in range(L):
        H_x = H_x + sx_list[i]
    H = -J * H_zz - g * H_x
    E, V = eigsh(H, k=1, which='SA', return_eigenvectors=True, ncv=20)
    if return_psi:
        return E[0], V[:, 0]
    return E[0]


"""
By performing Jordan-Wigner, Fourier and Bogoliubov transformations, the TFI model with PBC can be
diagonalized analytically. The Hamiltonian in terms of fermionic creation and annihilation operators
\\gamma_{p}^{\\dagger} and \\gamma_{p} reads:

H = (\\sum_{p} \\epsilon(p) \\gamma_{p}^{\\dagger}\\gamma_{p}) + E0.

    - Single particle excitation energy: \\epsilon(p) = 2 \\sqrt{J^2 - 2Jg\\cos(p) + g^2}.

    - Ground state energy: E0 = -\\sum_{p} \\epsilon(p)/2. 
"""

def epsilon(p, J, g):
    return 2 * np.sqrt(J**2 - 2 * J * g * np.cos(p) + g**2)

def infinite_gs_energy(J, g):
    """For comparison: Calculate ground state energy density from analytic formula.
    
    Compared to the above formula, we replace sum_k -> integral dk/2pi, to obtain the ground state 
    energy density in the thermodynamic limit.
    """
    e0_exact = -1 / (2 * np.pi) * scipy.integrate.quad(epsilon, -np.pi, np.pi, args=(J, g))[0]/2
    return e0_exact

def infinite_excitation_dispersion(J, g):
    """For comparison: Calculate excitation dispersion relation from analytic formula."""
    ps = np.arange(-np.pi, np.pi, 0.01)
    es_exact = epsilon(ps, J, g)
    return ps, es_exact