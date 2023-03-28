r"""Provides exact entropies for time evolution of free fermion models."""
# Copyright 2021 TeNPy Developers, GNU GPLv3

import numpy as np
import matplotlib.pyplot as plt


def hopping_matrix(L, t=1., mu_staggered=0., boundary_conditions='open'):
    r"""Generate matrix for fermionic hopping in real-space c_i operators.

    Provide hopping matrix h_{ij} for

    .. math ::
        H = \sum_{ij} h_{ij} c_i^\dagger c_j
          = t \sum_{i} (c_i^\dagger c^\j + h.c.)  + \mu_s (-1)^{i} n_i

    for open boundary conditions on L sites.
    """
    assert boundary_conditions in ['open', 'periodic']
    H = np.zeros((L,L))
    for i in range(L - int(boundary_conditions == 'open')):
        H[i, (i+1) % L] = t
        H[(i+1) % L, i] = t
    for i in range(L):
        H[i, i] =  mu_staggered*(-1)**i
    return H


def charge_density_wave(L):
    """Initial state |010101010...>"""
    psi0 = np.mod(np.arange(0,L),2)
    return np.diag(psi0)


def time_evolved_state(H_hop, psi0, time_list):
    """Evolve an initial correlation matrix with H_hop."""
    E, U = np.linalg.eigh(H_hop)
    for time in time_list:
        X = np.dot(np.dot(np.conj(U),np.diag(np.exp(1j*time*E))),U.T)
        psi_t = np.dot(X,np.dot(psi0, np.conj(X.T)))
        yield psi_t  # c^dagger_i c_j


def entanglement_entropy(psi, bond=None):
    """Calculate entanglement entropy for cutting at given bond (default: half-chain)."""
    L = psi.shape[0]
    if bond is None:
        bond = L // 2
    z = np.linalg.eigvalsh(psi[:bond, :bond])
    z = z[np.logical_and(z > 1.e-15, z < 1. - 1.e-15)]
    S = - np.sum(z * np.log(z) + (1. - z) * np.log(1. - z))
    return S


def XX_model_ground_state_energy(L, h_staggered, boundary_conditions='open'):
    """
    """
    H_hop = hopping_matrix(L, 2., 2.*h_staggered, boundary_conditions=boundary_conditions)
    E = np.linalg.eigvalsh(H_hop)
    E_shift = 0. if L % 2 == 0 else - h_staggered
    # the shift stems from the 1/2 in mapping sigmaz = (n - 1/2) for the h_s terms
    # which cancels out for even L due to the alternating sign of h_s
    return np.sum(E[:L//2]) + E_shift

def XX_model_time_evolved_entropies(L, h_staggered, time_list, bond=None,
                                    boundary_conditions='open'):
    r"""Half-chain entanglement entropies for time evolving Neel state with XX chain.

    The XX chain given by the hamiltonian (here, X,Y,Z = Pauli matrices)

    .. math ::
        H = \sum_{i} (X_i X_{i+1} + Y_i Y_{i+1}) - h_s (-1)^i Z_i

    maps to free fermions through the Jordan-Wigner transformation.
    This function returns the entropies for a time evolution starting from the Neel state
    |up down up down ...>.
    """
    psi_0 = charge_density_wave(L)
    H_hop = hopping_matrix(L, 2., h_staggered, boundary_conditions=boundary_conditions)
    S_list = []
    for psi_t in time_evolved_state(H_hop, psi_0, time_list):
        S_list.append(entanglement_entropy(psi_t))
    return np.array(S_list)


if __name__ == "__main__":
    L = 100
    h_staggered = 1.
    time_list = np.linspace(0, 20., 100)
    for h_s in [0., 1., 2.]:
        S_list = XX_model_comparison_entropies(L, h_s, time_list)
        plt.plot(time_list, S_list, label="$h_s = {h_s:1.1f}$".format(h_s=h_s))
    h_s = 0.
    S_list = XX_model_comparison_entropies(L, h_s, time_list,
                                           boundary_conditions='periodic')
    plt.plot(time_list, S_list, linestyle='--',
             label="$h_s = {h_s:1.1f}$, periodic".format(h_s=h_s))
    S_list = XX_model_comparison_entropies(10, h_s, time_list)
    plt.plot(time_list, S_list, linestyle=':',
             label="$h_s = {h_s:1.1f}, L=10$".format(h_s=h_s))

    plt.xlabel('$t$')
    plt.ylabel('$S$')
    plt.legend(loc='best')
    plt.show()
