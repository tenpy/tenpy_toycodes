"""Toy code implementing the uniform time dependent variational principle (uTDVP).

This implementation closely follows Laurens Vanderstraeten, Jutho Haegeman and Frank Verstraete, 
Tangent-space methods for uniform matrix product states, SciPost Physics Lecture Notes 007, 2019, 
https://arxiv.org/abs/1810.07006.
"""

import numpy as np
from scipy.sparse.linalg import expm_multiply

from .f_umps import UniformMPS
from .g_vumps import Heff1, Heff0, get_Lh, get_Rh, subtract_energy_offset, get_AL_AR


def utdvp_algorithm(psi0, h, dt, T):
    """Evolve the uMPS psi0 according to the Hamiltonian h up to time T in steps of dt."""
    tdvp_engine = UTDVPEngine(psi0, h, tol=1.e-10)
    ts = []
    Ss = []  # measure the entanglement entropy (or any other observable)
    t = 0.
    ts.append(t)
    Ss.append(psi0.get_entanglement_entropy())
    N_steps = int(T/dt + 0.5)
    for i in range(N_steps):
        tdvp_engine.run(dt)
        t += dt
        ts.append(t)
        Ss.append(tdvp_engine.psi.get_entanglement_entropy())
    print(f"uMPS evolved with TDVP up to time T={T} in steps of dt={dt}.")
    return ts, Ss


class UTDVPEngine:
    """Simple class for the uTDVP engine to evolve uMPS psi according to h for small time step dt.

    Approximately integrates (d/dt)|psi(A)> = (-i * P_A * H)|psi(A)>, where P_A is the tangent-space
    projector and H = sum_n h_{n,n+1}.
    
    Parameters
    ----------
    psi, h, tol: Same as attributes.

    Attributes
    ----------
    psi: UniformMPS
         The current state to be time-evolved.
    h: np.array[ndim=4]
       The two-site Hamiltonian governing the time-evolution.
    tol: float
         Tolerance up to which the geometric sum environments Lh and Rh are computed with gmres.
    Lh: np.array[ndim=2]
        Left environment computed from geometric sum of transfer matrix TL.
    Rh: np.array[ndim=2]
        Right environment computed from geometric sum of transfer matrix TR.
    D, d: int
          The bond dimension and physical dimension of psi.
    """
    def __init__(self, psi, h, tol):
        self.psi = psi.copy()
        self.h = subtract_energy_offset(self.psi, h, canonical_form=False)
        self.tol = tol
        self.Lh = get_Lh(self.psi, self.h, canonical_form=False, guess=None, tol=self.tol)
        self.Rh = get_Rh(self.psi, self.h, canonical_form=False, guess=None, tol=self.tol)
        
    def run(self, dt):
        """Evolve self.psi according to self.h for small time step dt.

        AC -> exp(-i * dt * Heff1)AC,
        C -> exp(-i * dt * Heff0)C,
        AL/AR from left/right polar decompositions of AC and C.
        """
        H_eff_1 = Heff1(self.h, self.Lh, self.psi.AL, self.psi.AR, self.Rh)
        H_eff_0 = Heff0(self.h, self.Lh, self.psi.AL, self.psi.AR, self.Rh)
        AC_new = self.evolve_theta(Heff=H_eff_1, theta=self.psi.AC, dt=dt)  
        C_new = self.evolve_theta(Heff=H_eff_0, theta=self.psi.C, dt=dt)  
        AL_new, AR_new = get_AL_AR(AC_new, C_new)  
        self.psi = UniformMPS(AL_new, AR_new, AC_new, C_new)
        self.h = subtract_energy_offset(self.psi, self.h, canonical_form=False)
        self.Lh = get_Lh(self.psi, self.h, canonical_form=False, guess=self.Lh, tol=self.tol)
        self.Rh = get_Rh(self.psi, self.h, canonical_form=False, guess=self.Rh, tol=self.tol)

    @staticmethod
    def evolve_theta(Heff, theta, dt):
        """Evolve |theta> -> exp(-i * dt * Heff)|theta>."""
        theta = np.reshape(theta, Heff.shape[1])
        theta_new = expm_multiply(-1.j * dt * Heff, theta, traceA=-1.j*dt*Heff.trace())
        theta_new /= np.linalg.norm(theta_new)
        theta_new = np.reshape(theta_new, Heff.shape_theta)
        return theta_new