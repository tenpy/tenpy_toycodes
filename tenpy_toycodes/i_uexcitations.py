"""Toy code implementing variational plane waxe excitations on top of a uMPS ground state.

This implementation closely follows Laurens Vanderstraeten, Jutho Haegeman and Frank Verstraete, 
Tangent-space methods for uniform matrix product states, SciPost Physics Lecture Notes 007, 2019, 
https://arxiv.org/abs/1810.07006.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import null_space

from tenpy_toycodes.f_umps import TransferMatrix
from tenpy_toycodes.g_vumps import InverseGeometricSum, get_Lh, get_Rh, subtract_energy_offset


class VariationalPlaneWaveExcitationEngine:
    """Simple class for variationally finding plane wave excitations on top of a uMPS ground state.

    ansatz: |phi(p,X; A)> = sum_n e^{i*p*n} ...--(AL)--(AL)--(VL)-(X)--(AR)--(AR)--...,
                                                  |     |     |         |     |

    where p is the momentum, and X the left-gauge parametrization of the tensor 
    
        --(B)-- = --(VL)-(X)--, 
           |         | 

    perturbing the ground state |psi(A)> around site n. 

    With <phi(p',X'; A)|H|phi(p,X; A)> = 2 * pi * delta(p-p') * <X'|Heff(p)|X>, the variational
    optimization boils down to diagonalizing Heff(p) for a few lowest-lying eigenvalues.

    For a two-fold degnerate, symmetry-broken ground state, topological domain wall excitations can
    be targeted by taking AL and AR from the two orthogonal ground states. To fix the momentum 
    unambiguously, we rescale AR with a phase, such that the mixed transfer matrix has a positive 
    leading eigenvalue.

    Parameters
    ----------
    psi0: UniformMPS
          The ground state on top of which the excited states are searched.
    h, p, k, tol: Same as attributes.
    psi0_tilde: UniformMPS or None
                If not None, second degnerate, symmetry-broken ground state.

    Attributes
    ----------
    h: np.array[ndim=4]
       The two-site Hamiltonian of which the excitations are searched.
    p: float
       Momentum value between -pi and pi.
    k: int
       The number of excitations to be computed (only a few lowest-lying have physical meaning).
    tol: float
         The tolerance up to which geometric sum environments are computed with gmres.
    VL: np.array[ndim=3]
        Left-gauge tensor with legs vL p vvR of dimension D x d x D*(d-1).
    AL: np.array[ndim=3]
        Left orthonormal tensor of the ground state.
    Lh: np.array[ndim=2]
        Left environment computed from geometric sum of transfer matrix TL.
    AR: np.array[ndim=3]
        Right orthonormal tensor of the (second degenerate) ground state.
    Rh: np.array[ndim=2]
        Right environment computed from geometric sum of transfer matrix TR.
    IGS_RL: InverseGeometricSum
            Inverse geometric sum of the mixed transfer matrix of AR and AL.
    IGS_LR: InverseGeometricSum
            Inverse geometric sum of the mixed transfer matrix of AL and AR.
    """
    def __init__(self, psi0, h, p, k, tol, psi0_tilde=None):
        self.h = subtract_energy_offset(psi0, h, canonical_form=True)
        self.p = p
        self.k = k
        self.tol = tol
        self.VL = self.get_VL(psi0.AL)
        self.AL = psi0.AL
        self.Lh = get_Lh(psi0, self.h, canonical_form=True, guess=None, tol=self.tol)
        if psi0_tilde is None:
            self.AR = psi0.AR
            self.Rh = get_Rh(psi0, self.h, canonical_form=True, guess=None, tol=self.tol)
            self.IGS_RL = InverseGeometricSum(psi0.AR, psi0.AL, R=np.conj(psi0.C).T, L=psi0.C.T, \
                                              transpose=True, alpha=np.exp(-1.j * p), pseudo=True)
            self.IGS_LR = InverseGeometricSum(psi0.AL, psi0.AR, R=psi0.C, L=np.conj(psi0.C), \
                                              transpose=False, alpha=np.exp(1.j * p), pseudo=True)
        else:
            AR = self.fix_momentum(psi0.AL, psi0_tilde.AR)
            psi0_tilde.AR = self.AR = AR
            self.Rh = get_Rh(psi0_tilde, self.h, canonical_form=True, guess=None, tol=self.tol)
            self.IGS_RL = InverseGeometricSum(psi0_tilde.AR, psi0.AL, R=None, L=None, \
                                              transpose=True, alpha=np.exp(-1.j * p), pseudo=False)
            self.IGS_LR = InverseGeometricSum(psi0.AL, psi0_tilde.AR, R=None, L=None, \
                                              transpose=False, alpha=np.exp(1.j * p), pseudo=False)
            
    def run(self):
        """For one momentum value self.p, compute self.k excitations."""
        H_eff = Heff(self.h, self.p, self.VL, self.AL, self.AR, self.Lh, self.Rh, \
                     self.IGS_RL, self.IGS_LR, self.tol)
        es, Xs = eigsh(H_eff, k=self.k, which="SA")
        Xs_matrices = []
        for i in range(self.k):
            X = Xs[:, i]
            Xs_matrices.append(np.reshape(X, H_eff.shape_X))
        if self.k == 1:
            return es[0], Xs_matrices[0]        
        return es, Xs_matrices

    @staticmethod
    def get_VL(AL):
        """For left orthonormal tensor AL, compute tensor VL of dimension D x d x D*(d-1), such that
    
        .--(VL)--vvR
        |   |                                  
        |   |        =  0   <->   vR*--(AL^{dagger})----(VL)--vvR  =  0.
        |   |                                |           |          
        .-(AL*)--vR*                         .-----------.

        Interpreting AL as the first D orthonormal columns of a (D*d)x(D*d) unitary matrix, 
        VL corresponds to the remaining D*(d-1) columns thereof.
        """
        D = np.shape(AL)[0]
        d = np.shape(AL)[1]
        AL = np.reshape(AL, (D * d, D))  # vL.p vR
        VL = null_space(np.conj(AL).T)  # vL.p vvR
        VL = np.reshape(VL, (D, d, D * (d-1)))  # vL p vvR
        return VL
    
    @staticmethod
    def fix_momentum(AL, AR):
        """Multiply AR with a phase such that TRL has a positive leading eigenvalue."""
        lambda1, _ = TransferMatrix([AR], [AL]).get_leading_eigenpairs(k=1)
        AR *= np.conj(lambda1)/np.abs(lambda1)
        return AR
    

class Heff(LinearOperator):
    """Class for the effective Hamiltonian acting on the parametrization X of the perturbation B.

                             
                                .---(B)---                      .---(B)---  
                                |    |                          |    |
    --(B)--  =  --(VL)-(X)--,   |    |      =  0,   --(X)--  =  |    |   
       |           |            |    |                          |    |
                                .--(AL*)--                      .--(VL*)--

                                
                                              ...---(AL)---(AL)---(B)---(AR)---(AR)---...
                                                     |      |      |     |      |
                                                                   pm

                                                                  pn* p(n+1)*
                                                     |      |      |     |      |
    matvec:  --(B)--  ->  sum_{n,m} e^{i*p*m} ...    |      |    (----h----)    |     ...
                |                                    |      |      |     |      |
                                                                   pn  p(n+1)
                                                    
                                                     |      |      |     |      |              
                                              ...--(AL*)--(AL*)--    --(AR*)--(AR*)---...
                                                                 
    """
    def __init__(self, h, p, VL, AL, AR, Lh, Rh, IGS_RL, IGS_LR, tol):
        self.h = h
        self.p = p
        self.VL = VL
        self.AL = AL
        self.AR = AR
        self.Lh = Lh
        self.Rh = Rh
        self.IGS_RL = IGS_RL
        self.IGS_LR = IGS_LR
        self.tol = tol
        D = np.shape(self.AL)[0]
        d = np.shape(self.AL)[1]
        self.shape = (D * (d-1) * D, D * (d-1) * D)
        self.shape_X = (D * (d-1), D)
        self.dtype = self.AL.dtype

    def _matvec(self, X):
        """Perform the matvec multiplication diagrammatically shown above."""
        X = np.reshape(X, self.shape_X)  # vvL vR
        B = np.tensordot(self.VL, X, axes=(2, 0))  # vL p [vvR], [vvL] vR
        """m < 0"""
        "n < -1"
        lB = np.tensordot(self.AL, B, axes=(2, 0))  # vL p1 [vR], [vL] p2 vR
        lB = np.tensordot(self.h, lB, axes=((2, 3), (1, 2)))  # p1 p2 [p1*] [p2*], vL [p1] [p2] vR
        lB = np.tensordot(np.conj(self.AL), lB, axes=((0, 1), (2, 0)))  
        # [vL*] [p1*] vR*, [p1] p2 [vL] vR
        lB = np.tensordot(lB, np.conj(self.AL), axes=((0, 1), (0, 1)))  
        # [vR*] [p2] vR, [vL*] [p2*] vR* 
        LB = self.IGS_RL.multiply_geometric_sum(lB, guess=None, tol=self.tol)  # vR vR*
        B_new = np.exp(-1.j * self.p) * np.tensordot(LB, self.AR, axes=(0, 0))  
        # [vR] vR*, [vL] p vR -> vL p vR
        lB = np.tensordot(B, self.AR, axes=(2, 0))  # vL p1 [vR], [vL] p2 vR
        lB = np.tensordot(self.h, lB, axes=((2, 3), (1, 2)))  # p1 p2 [p1*] [p2*], vL [p1] [p2] vR
        lB = np.tensordot(np.conj(self.AL), lB, axes=((0, 1), (2, 0)))  
        # [vL*] [p1*] vR*, [p1] p2 [vL] vR
        lB = np.tensordot(lB, np.conj(self.AL), axes=((0, 1), (0, 1)))  
        # [vR*] [p2] vR, [vL*] [p2*] vR* 
        LB = self.IGS_RL.multiply_geometric_sum(lB, guess=None, tol=self.tol)  # vR vR*
        B_new += np.exp(-2.j * self.p) * np.tensordot(LB, self.AR, axes=(0, 0))  
        # [vR] vR*, [vL] p vR -> vL p vR
        lB = np.tensordot(B, self.Lh, axes=(0, 0)) # [vL] p vR, [vR] vR*
        lB = np.tensordot(lB, np.conj(self.AL), axes=((0, 2), (1, 0)))  
        # [p] vR [vR*], [vL*] [p*] vR*
        LB = self.IGS_RL.multiply_geometric_sum(lB, guess=None, tol=self.tol)  # vR vR*
        B_new += np.exp(-1.j * self.p) * np.tensordot(LB, self.AR, axes=(0, 0))  
        # [vR] vR*, [vL] p vR -> vL p vR
        "n = -1"
        b = np.tensordot(B, self.AR, axes=(2, 0))  # vL p1 [vR], [vL] p2 vR
        b = np.tensordot(self.h, b, axes=((2, 3), (1, 2)))  # p1 p2 [p1*] [p2*], vL [p1] [p2] vR
        b = np.tensordot(np.conj(self.AL), b, axes=((0, 1), (2, 0)))  
        # [vL*] [p1*] vR*, [p1] p2 [vL] vR
        B_new += np.exp(-1.j * self.p) * b  # vL p vR
        "n = 0: no contribution in left gauge" 
        "n > 0: no contribution in left gauge"
        """m = 0"""
        "n < -1"
        B_new += np.tensordot(self.Lh, B, axes=(0, 0))  # [vR] vR*, [vL] p vR -> vL p vR
        "n = -1"
        b = np.tensordot(self.AL, B, axes=(2, 0))  # vL p1 [vR], [vL] p2 vR
        b = np.tensordot(self.h, b, axes=((2, 3), (1, 2)))  # p1 p2 [p1*] [p2*], vL [p1] [p2] vR
        b = np.tensordot(np.conj(self.AL), b, axes=((0, 1), (2, 0)))  
        # [vL*] [p1*] vR*, [p1] p2 [vL] vR
        B_new += b  # vL p vR
        "n = 0"
        b = np.tensordot(B, self.AR, axes=(2, 0))  # vL p1 [vR], [vL] p2 vR
        b = np.tensordot(self.h, b, axes=((2, 3), (1, 2)))  # p1 p2 [p1*] [p2*], vL [p1] [p2] vR
        b = np.tensordot(b, np.conj(self.AR), axes=((1, 3), (1, 2)))  
        # p1 [p2] vL [vR], vL* [p2*] [vR*]
        B_new += np.transpose(b, (1, 0, 2))  # vL p vR
        "n > 0"
        B_new += np.tensordot(B, self.Rh, axes=(2, 0))  # vL p [vR], [vL] vL* -> vL p vR
        """m > 0"""
        "n < -1"
        rB = np.tensordot(B, np.conj(self.AR), axes=((1, 2), (1, 2)))  # vL [p] [vR], vL* [p*] [vR*]
        RB = self.IGS_LR.multiply_geometric_sum(rB, guess=None, tol=self.tol)  # vL vL*
        b = np.tensordot(self.AL, RB, axes=(2, 0))  # vL p [vR], [vL] vL*
        b = np.tensordot(self.Lh, b, axes=(0, 0))  # [vR] vR*, [vL] p vL*
        B_new += np.exp(1.j * self.p) * b  # vL p vR
        "n = -1"
        b = np.tensordot(self.AL, self.AL, axes=((2, 0)))  # vL p1 [vR], [vL] p2 vR
        b = np.tensordot(self.h, b, axes=((2, 3), (1, 2)))  # p1 p2 [p1*] [p2*], vL [p1] [p2] vR
        b = np.tensordot(np.conj(self.AL), b, axes=((0, 1), (2, 0)))  
        # [vL*] [p1*] vR*, [p1] p2 [vL] vR
        B_new += np.exp(1.j * self.p) * np.tensordot(b, RB, axes=(2, 0))  
        # vR* p2 [vR], [vL] vL* -> vL p vR
        "n = 0"
        b = np.tensordot(self.AL, self.AL, axes=((2, 0)))  # vL p1 [vR], [vL] p2 vR
        b = np.tensordot(self.h, b, axes=((2, 3), (1, 2)))  # p1 p2 [p1*] [p2*], vL [p1] [p2] vR
        b = np.tensordot(b, RB, axes=(3, 0))  # p1 p2 vL [vR], [vL] vL*
        b = np.tensordot(b, np.conj(self.AR), axes=((1, 3), (1, 2)))  
        # p1 [p2] vL [vL*], vL* [p2*] [vR*]
        B_new += np.exp(2.j * self.p) * np.transpose(b, (1, 0, 2))  # vL p vR
        b = np.tensordot(self.AL, B, axes=(2, 0))  # vL p1 [vR], [vL] p2 vR
        b = np.tensordot(self.h, b, axes=((2, 3), (1, 2)))  # p1 p2 [p1*] [p2*], vL [p1] [p2] vR
        b = np.tensordot(b, np.conj(self.AR), axes=((1, 3), (1, 2)))  
        # p1 [p2] vL [vR], vL* [p2*] [vR*]
        B_new += np.exp(1.j * self.p) * np.transpose(b, (1, 0, 2))  # vL p vR            
        "n > 0"
        rb = np.tensordot(B, self.AR, axes=(2, 0))  # vL p1 [vR], [vL] p2 vR
        rb = np.tensordot(self.h, rb, axes=((2, 3), (1, 2)))  # p1 p2 [p1*] [p2*], vL [p1] [p2] vR
        rb = np.tensordot(rb, np.conj(self.AR), axes=((1, 3), (1, 2)))  
        # p1 [p2] vL [vR], vL* [p2*] [vR*]
        rb = np.tensordot(rb, np.conj(self.AR), axes=((0, 2), (1, 2)))  
        # [p1] vL [vL*], vL* [p1*] [vR*]
        RB = self.IGS_LR.multiply_geometric_sum(rb, guess=None, tol=self.tol)  # vL vL*
        B_new += np.exp(1.j * self.p) * np.tensordot(self.AL, RB, axes=(2, 0))  
        # vL p [vR], [vL] vL* -> vL p vR
        rb = np.tensordot(self.AL, B, axes=(2, 0))  # vL p1 [vR], [vL] p2 vR
        rb = np.tensordot(self.h, rb, axes=((2, 3), (1, 2)))  # p1 p2 [p1*] [p2*], vL [p1] [p2] vR
        rb = np.tensordot(rb, np.conj(self.AR), axes=((1, 3), (1, 2)))  
        # p1 [p2] vL [vR], vL* [p2*] [vR*]
        rb = np.tensordot(rb, np.conj(self.AR), axes=((0, 2), (1, 2)))  
        # [p1] vL [vL*], vL* [p1*] [vR*]
        RB = self.IGS_LR.multiply_geometric_sum(rb, guess=None, tol=self.tol)  # vL vL*
        B_new += np.exp(2.j * self.p) * np.tensordot(self.AL, RB, axes=(2, 0))  
        # vL p [vR], [vL] vL* -> vL p vR
        rB = np.tensordot(B, np.conj(self.AR), axes=((1, 2), (1, 2)))  # vL [p] [vR], vL* [p*] [vR*]
        RB = self.IGS_LR.multiply_geometric_sum(rB, guess=None, tol=self.tol)  # vL vL*
        rb = np.tensordot(self.AL, self.AL, axes=(2, 0))  # vL p1 [vR], [vL] p2 vR
        rb = np.tensordot(self.h, rb, axes=((2, 3), (1, 2)))  # p1 p2 [p1*] [p2*], vL [p1] [p2] vR
        rb = np.tensordot(rb, RB, axes=(3, 0))  # p1 p2 vL [vR], [vL] vL*
        rb = np.tensordot(rb, np.conj(self.AR), axes=((1, 3), (1, 2)))  
        # p1 [p2] vL [vL*], vL* [p2*] [vR*]
        rb = np.tensordot(rb, np.conj(self.AR), axes=((0, 2), (1, 2)))  
        # [p1] vL [vL*], vL* [p1*] [vR*]
        RB = self.IGS_LR.multiply_geometric_sum(rb, guess=None, tol=self.tol)  # vL vL*
        B_new += np.exp(3.j * self.p) * np.tensordot(self.AL, RB, axes=(2, 0))  
        # vL p [vR], [vL] vL* -> vL p vR
        rb = np.tensordot(B, self.Rh, axes=(2, 0))  # vL p [vR], [vL] vL*
        rb = np.tensordot(rb, np.conj(self.AR), axes=((1, 2), (1, 2)))  
        # vL [p] [vL*], vL* [p*] [vR*]
        RB = self.IGS_LR.multiply_geometric_sum(rb, guess=None, tol=self.tol)  # vL vL*
        B_new += np.exp(1.j * self.p) * np.tensordot(self.AL, RB, axes=(2, 0))  
        # vL p [vR], [vL] vL* -> vL p vR
        X_new = np.tensordot(np.conj(self.VL), B_new, axes=((0, 1), (0, 1)))  
        # [vL*] [p*] vvR*, [vL] [p] vR
        X_new = np.reshape(X_new, self.shape[1])  # vvL.vR
        return X_new