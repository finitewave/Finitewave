import numpy as np
from numba import njit, prange

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import (
    AsymmetricStencil2D
)
from finitewave.cpuwave2D.stencil.isotropic_stencil_2d import (
    IsotropicStencil2D
)


class FentonKarma2D(CardiacModel):

    def __init__(self):
        """
        Two-dimensional implementation of the Fenton-Karma model of cardiac electrophysiology.

        The Fenton-Karma model is a minimal three-variable model designed to reproduce
        essential features of human ventricular action potentials, including restitution, 
        conduction velocity dynamics, and spiral wave behavior. It captures the interaction 
        between fast depolarization, slow repolarization, and calcium-mediated effects 
        through simplified phenomenological equations.

        This implementation corresponds to the MLR-I parameter set described in the original paper
        and supports 2D isotropic and anisotropic tissue simulations with diffusion.

        Attributes
        ----------
        u : np.ndarray
            Transmembrane potential (normalized, dimensionless).
        v : np.ndarray
            Fast recovery variable, representing sodium channel inactivation.
        w : np.ndarray
            Slow recovery variable, representing calcium channel dynamics.
        D_model : float
            Baseline diffusion coefficient used in the diffusion stencil.
        state_vars : list of str
            Names of the state variables stored during the simulation.
        npfloat : str
            Floating point precision (default is 'float64').

        Model Parameters
        ----------------
        tau_r : float
            Time constant for repolarization (outward current).
        tau_o : float
            Time constant for the open-state decay of fast sodium channels.
        tau_d : float
            Time constant for depolarization (fast inward current).
        tau_si : float
            Time constant for the slow inward (calcium-like) current.
        tau_v_m : float
            Time constant for inactivation gate v (membrane below threshold).
        tau_v_p : float
            Time constant for recovery gate v (above threshold).
        tau_w_m : float
            Time constant for recovery gate w (below threshold).
        tau_w_p : float
            Time constant for decay of w (above threshold).
        k : float
            Steepness parameter for the slow inward current.
        u_c : float
            Activation threshold for recovery dynamics.
        uc_si : float
            Activation threshold for the slow inward current.
        
        Paper
        -----
        Fenton, F., & Karma, A. (1998).
        Vortex dynamics in three-dimensional continuous myocardium 
        with fiber rotation: Filament instability and fibrillation.
        Chaos, 8(1), 20-47.
        https://doi.org/10.1063/1.166311
                
        """
        super().__init__()
        self.v = np.ndarray
        self.w = np.ndarray

        self.D_model = 1.

        self.state_vars = ["u", "v", "w"]
        self.npfloat    = 'float64'

        # model parameters (MLR-I)
        
        self.tau_r   = 130
        self.tau_o   = 12.5
        self.tau_d   = 0.172
        self.tau_si  = 127
        self.tau_v_m = 18.2
        self.tau_v_p = 10
        self.tau_w_m = 80
        self.tau_w_p = 1020
        self.k       = 10
        self.u_c     = 0.13
        self.uc_si   = 0.85


    def initialize(self):
        """
        Initializes the model for simulation.
        """
        super().initialize()
        self.v = np.ones_like(self.u, dtype=self.npfloat)
        self.w = np.ones_like(self.u, dtype=self.npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel for the Fenton-Karma model.
        """
        ionic_kernel_2d(self.u_new, self.u, self.v, self.w, self.cardiac_tissue.myo_indexes, 
                        self.dt, self.tau_d, self.tau_o, self.tau_r, self.tau_si, 
                        self.tau_v_m, self.tau_v_p, self.tau_w_m, self.tau_w_p,
                        self.k, self.u_c, self.uc_si)

    def select_stencil(self, cardiac_tissue):
        """
        Selects the appropriate stencil for diffusion based on the tissue
        properties. If the tissue has fiber directions, an asymmetric stencil
        is used; otherwise, an isotropic stencil is used.

        Parameters
        ----------
        cardiac_tissue : CardiacTissue2D
            A tissue object representing the cardiac tissue.

        Returns
        -------
        Stencil
            The stencil object to use for diffusion computations.
        """
        if cardiac_tissue.fibers is None:
            return IsotropicStencil2D()

        return AsymmetricStencil2D()
    
@njit
def calc_Jfi(u, v, u_c, tau_d):
    """
    Computes the fast inward current (J_fi) for the Fenton-Karma model.

    This current is responsible for the rapid depolarization of the membrane
    potential. It is active only when the membrane potential exceeds a threshold `u_c`.

    Parameters
    ----------
    u : float
        Current membrane potential (dimensionless).
    v : float
        Fast recovery gate (sodium channel inactivation).
    u_c : float
        Activation threshold for the inward current.
    tau_d : float
        Time constant for depolarization.

    Returns
    -------
    float
        Value of the fast inward current at this point.
    """
    H = 1.0 if (u - u_c) >= 0 else 0.0
    return -(v*H*(1-u)*(u - u_c))/tau_d

@njit
def calc_Jso(u, u_c, tau_o, tau_r):
    """
    Computes the slow outward current (J_so) for repolarization.

    This current contains two parts:
    - a linear repolarizing component active below threshold `u_c`
    - a constant repolarizing component above threshold

    Parameters
    ----------
    u : float
        Membrane potential.
    u_c : float
        Activation threshold.
    tau_o : float
        Time constant for subthreshold repolarization.
    tau_r : float
        Time constant for suprathreshold repolarization.

    Returns
    -------
    float
        Value of the outward repolarizing current.
    """
    H1 = 1.0 if (u_c - u) >= 0 else 0.0
    H2 = 1.0 if (u - u_c) >= 0 else 0.0

    return u*H1/tau_o + H2/tau_r

@njit
def calc_Jsi(u, w, k, uc_si, tau_si):
    """
    Computes the slow inward (calcium-like) current (J_si).

    This current is responsible for the plateau phase of the action potential
    and depends on the gating variable `w` and a smoothed activation threshold.

    Parameters
    ----------
    u : float
        Membrane potential.
    w : float
        Slow recovery gate.
    k : float
        Steepness of the tanh activation curve.
    uc_si : float
        Activation threshold for the slow inward current.
    tau_si : float
        Time constant for the slow inward current.

    Returns
    -------
    float
        Value of the slow inward current.
    """
    return -w*(1 + np.tanh(k*(u - uc_si)))/(2*tau_si)

@njit
def calc_v(v, u, dt, u_c, tau_v_m, tau_v_p):
    """
    Updates the fast recovery gate `v` over time.

    This gate controls sodium channel availability and changes depending on
    whether the membrane potential is below or above a critical threshold.

    Parameters
    ----------
    v : float
        Current value of the recovery variable.
    u : float
        Membrane potential.
    dt : float
        Time step.
    u_c : float
        Activation threshold.
    tau_v_m : float
        Time constant below threshold.
    tau_v_p : float
        Time constant above threshold.

    Returns
    -------
    float
        Updated value of `v`.
    """
    H1 = 1.0 if (u_c - u) >= 0 else 0.0
    H2 = 1.0 if (u - u_c) >= 0 else 0.0
    v += dt*(H1*(1 - v)/tau_v_m - H2*v/tau_v_p)
    return v

@njit
def calc_w(w, u, dt, u_c, tau_w_m, tau_w_p):
    """
    Updates the slow recovery gate `w` over time.

    This gate represents the calcium channel recovery and decays similarly to `v`,
    depending on whether the membrane potential is above or below threshold `u_c`.

    Parameters
    ----------
    w : float
        Current value of the recovery variable.
    u : float
        Membrane potential.
    dt : float
        Time step.
    u_c : float
        Activation threshold.
    tau_w_m : float
        Time constant below threshold.
    tau_w_p : float
        Time constant above threshold.

    Returns
    -------
    float
        Updated value of `w`.
    """
    H1 = 1.0 if (u_c - u) >= 0 else 0.0
    H2 = 1.0 if (u - u_c) >= 0 else 0.0
    w += dt*(H1*(1 - w)/tau_w_m - H2*w/tau_w_p)
    return w

@njit(parallel=True)
def ionic_kernel_2d(u_new, u, v, w, indexes, dt, 
                    tau_d, tau_o, tau_r, tau_si, 
                    tau_v_m, tau_v_p, tau_w_m, tau_w_p,
                    k, u_c, uc_si):
    """
    Computes the ionic kernel for the Fenton-Karma 2D model.

    Parameters
    ----------
    u_new : np.ndarray
        Array to store the updated action potential values.
    u : np.ndarray
        Current action potential array.
    indexes : np.ndarray
        Array of indices where the kernel should be computed (``mesh == 1``).
    dt : float
        Time step for the simulation.
    """

    n_j = u.shape[1]

    for ind in prange(len(indexes)):
        ii = indexes[ind]
        i = int(ii / n_j)
        j = ii % n_j

        v[i, j] = calc_v(v[i, j], u[i, j], dt, u_c, tau_v_m, tau_v_p)
        w[i, j] = calc_w(w[i, j], u[i, j], dt, u_c, tau_w_m, tau_w_p)

        J_fi = calc_Jfi(u[i, j], v[i, j], u_c, tau_d)
        J_so = calc_Jso(u[i, j], u_c, tau_o, tau_r)
        J_si = calc_Jsi(u[i, j], v[i, j], k, uc_si, tau_si)

        u_new[i, j] += dt * (-J_fi - J_so - J_si)

