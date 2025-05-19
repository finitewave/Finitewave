import numpy as np
from numba import njit, prange

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import (
    AsymmetricStencil2D
)
from finitewave.cpuwave2D.stencil.isotropic_stencil_2d import (
    IsotropicStencil2D
)


class BuenoOrovio2D(CardiacModel):

    def __init__(self):
        """
        Two-dimensional implementation of the Bueno-Orovio–Cherry–Fenton (BOCF) model 
        for simulating human ventricular tissue electrophysiology.

        The BOCF model is a minimal phenomenological model developed to capture 
        key ionic mechanisms and reproduce realistic human ventricular action potential 
        dynamics, including restitution, conduction block, and spiral wave behavior. 
        It consists of four variables: transmembrane potential (u), two gating variables (v, w), 
        and one additional slow variable (s), representing calcium-related dynamics.

        This implementation corresponds to the EPI (epicardial) parameter set described in the paper.

        Attributes
        ----------
        u : np.ndarray
            Transmembrane potential (dimensionless, typically in [0, 1.55]).
        v : np.ndarray
            Fast gating variable representing sodium channel inactivation.
        w : np.ndarray
            Slow recovery variable representing calcium and potassium gating.
        s : np.ndarray
            Slow variable related to calcium inactivation.
        D_model : float
            Diffusion coefficient for spatial propagation.
        state_vars : list of str
            Names of state variables to be saved or restored.
        npfloat : str
            Floating point precision (default: 'float64').

        Model Parameters (EPI set)
        --------------------------
        u_o : float
            Resting membrane potential.
        u_u : float
            Peak potential (upper bound).
        theta_v, theta_w : float
            Activation thresholds for v and w.
        theta_v_m, theta_o : float
            Thresholds for switching time constants.
        tau_v1_m, tau_v2_m : float
            Time constants for v below/above threshold.
        tau_v_p : float
            Decay constant for v.
        tau_w1_m, tau_w2_m : float
            Base and transition time constants for w.
        k_w_m, u_w_m : float
            Parameters controlling the shape of τw curve.
        tau_w_p : float
            Time constant for decay of w above threshold.
        tau_fi : float
            Time constant for fast inward current (J_fi).
        tau_o1, tau_o2 : float
            Time constants for outward current below/above threshold.
        tau_so1, tau_so2 : float
            Time constants for repolarizing tail current.
        k_so, u_so : float
            Parameters controlling nonlinearity in tau_so.
        tau_s1, tau_s2 : float
            Time constants for the s-gate below/above threshold.
        k_s, u_s : float
            Parameters for tanh activation of the s variable.
        tau_si : float
            Time constant for slow inward current (J_si).
        tau_w_inf : float
            Slope of w∞ below threshold.
        w_inf_ : float
            Asymptotic value of w∞ above threshold.

        Paper
        -----
        Bueno-Orovio, A., Cherry, E. M., & Fenton, F. H. (2008).
        Minimal model for human ventricular action potentials in tissue.
        J Theor Biol., 253(3), 544-60.
        https://doi.org/10.1016/j.jtbi.2008.03.029

        """
        super().__init__()
        self.v = np.ndarray
        self.w = np.ndarray
        self.s = np.ndarray

        self.D_model = 1.

        self.state_vars = ["u", "v", "w", "s"]
        self.npfloat    = 'float64'

        # model parameters (EPI)
        
        self.u_o = 0.0
        self.u_u = 1.55
        self.theta_v = 0.3
        self.theta_w = 0.13
        self.theta_v_m = 0.006
        self.theta_o = 0.006
        self.tau_v1_m = 60
        self.tau_v2_m = 1150
        self.tau_v_p = 1.4506
        self.tau_w1_m = 60
        self.tau_w2_m = 15
        self.k_w_m = 65
        self.u_w_m = 0.03
        self.tau_w_p = 200
        self.tau_fi = 0.11
        self.tau_o1 = 400
        self.tau_o2 = 6
        self.tau_so1 = 30.0181
        self.tau_so2 = 0.9957
        self.k_so = 2.0458
        self.u_so = 0.65
        self.tau_s1 = 2.7342
        self.tau_s2 = 16
        self.k_s = 2.0994
        self.u_s = 0.9087
        self.tau_si = 1.8875
        self.tau_w_inf = 0.07
        self.w_inf_ = 0.94

        # initial conditions
        self.init_u = 0.0
        self.init_v = 1.0
        self.init_w = 1.0
        self.init_s = 0.0

    def initialize(self):
        """
        Initializes the model for simulation.
        """
        super().initialize()
        self.u = self.init_u * np.ones_like(self.u, dtype=self.npfloat)
        self.v = self.init_v * np.ones_like(self.u, dtype=self.npfloat) 
        self.w = self.init_w * np.ones_like(self.u, dtype=self.npfloat)
        self.s = self.init_s * np.ones_like(self.u, dtype=self.npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel for the Bueno-Orovio model.
        """
        ionic_kernel_2d(self.u_new, self.u, self.v, self.w, self.s, self.cardiac_tissue.myo_indexes, 
                        self.dt, self.u_o, self.u_u, self.theta_v, self.theta_w, self.theta_v_m,
                        self.theta_o, self.tau_v1_m, self.tau_v2_m, self.tau_v_p,
                        self.tau_w1_m, self.tau_w2_m, self.k_w_m, self.u_w_m,
                        self.tau_w_p, self.tau_fi, self.tau_o1, self.tau_o2,
                        self.tau_so1, self.tau_so2, self.k_so, self.u_so,
                        self.tau_s1, self.tau_s2, self.k_s, self.u_s,
                        self.tau_si, self.tau_w_inf, self.w_inf_)

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
def calc_v(v, u, dt, theta_v, v_inf, tau_v_m, tau_v_p):
    """
    Updates the fast inactivation gate variable `v`.

    The variable `v` models the fast sodium channel inactivation.
    It follows a piecewise ODE with different dynamics depending
    on whether the membrane potential `u` is above or below `theta_v`.

    Parameters
    ----------
    v : float
        Current value of the v gate.
    u : float
        Current membrane potential.
    dt : float
        Time step.
    theta_v : float
        Threshold for switching recovery behavior.
    v_inf : float
        Steady-state value of v.
    tau_v_m : float
        Time constant for activation (u < threshold).
    tau_v_p : float
        Time constant for decay (u >= threshold).

    Returns
    -------
    float
        Updated value of the v gate.
    """
    v_ = (v_inf - v)/tau_v_m if (u - theta_v) < 0 else -v/tau_v_p
    v += dt*v_
    return v

@njit
def calc_w(w, u, dt, theta_w, w_inf, tau_w_m, tau_w_p):
    """
    Updates the slow gating variable `w`.

    The variable `w` represents calcium/potassium channel gating.
    It has different recovery dynamics below and above the threshold `theta_w`.

    Parameters
    ----------
    w : float
        Current value of the w gate.
    u : float
        Membrane potential.
    dt : float
        Time step.
    theta_w : float
        Threshold for switching between time constants.
    w_inf : float
        Steady-state value of w.
    tau_w_m : float
        Time constant for approach to w_inf (u < threshold).
    tau_w_p : float
        Time constant for decay (u >= threshold).

    Returns
    -------
    float
        Updated value of the w gate.
    """
    w_ = (w_inf - w)/tau_w_m if (u - theta_w) < 0 else -w/tau_w_p
    w += dt*w_
    return w

@njit
def calc_s(s, u, dt, tau_s, k_s, u_s):
    """
    Updates the slow variable `s`, related to calcium dynamics.

    The variable `s` evolves toward a tanh-based steady-state function of `u`.

    Parameters
    ----------
    s : float
        Current value of the s variable.
    u : float
        Membrane potential.
    dt : float
        Time step.
    tau_s : float
        Time constant.
    k_s : float
        Slope of the tanh function.
    u_s : float
        Midpoint of the tanh function.

    Returns
    -------
    float
        Updated value of the s variable.
    """
    s += dt*((1 + np.tanh(k_s*(u - u_s)))/2 - s)/tau_s
    return s

@njit
def calc_Jfi(u, v, theta_v, u_u, tau_fi):
    """
    Computes the fast inward sodium current (J_fi).

    Active when membrane potential exceeds `theta_v`.
    Models rapid depolarization due to sodium influx.

    Parameters
    ----------
    u : float
        Membrane potential.
    v : float
        Fast gating variable.
    theta_v : float
        Activation threshold.
    u_u : float
        Upper limit for depolarization.
    tau_fi : float
        Time constant of the fast inward current.

    Returns
    -------
    float
        Current value of J_fi.
    """
    H = 1.0 if (u - theta_v) >= 0 else 0.0
    return -v*H*(u - theta_v)*(u_u - u)/tau_fi

@njit
def calc_Jso(u, u_o, theta_w, tau_o, tau_so):
    """
    Computes the slow outward current (J_so).

    Consists of a linear repolarization component below `theta_w`
    and a constant component above.

    Parameters
    ----------
    u : float
        Membrane potential.
    u_o : float
        Resting potential (offset).
    theta_w : float
        Threshold for switching between components.
    tau_o : float
        Time constant below threshold.
    tau_so : float
        Time constant above threshold.

    Returns
    -------
    float
        Current value of J_so.
    """
    H = 1.0 if (u - theta_w) >= 0 else 0.0
    return (u - u_o)*(1 - H)/tau_o + H/tau_so

@njit
def calc_Jsi(u, w, s, theta_w, tau_si):
    """
    Computes the slow inward current (J_si), active during plateau phase.

    Active only when `u > theta_w` and controlled by gating variables `w` and `s`.

    Parameters
    ----------
    u : float
        Membrane potential.
    w : float
        Slow gating variable.
    s : float
        Calcium-related variable.
    theta_w : float
        Threshold for activation.
    tau_si : float
        Time constant of slow inward current.

    Returns
    -------
    float
        Current value of J_si.
    """
    H = 1.0 if (u - theta_w) >= 0 else 0.0
    return -H*w*s/tau_si

@njit
def calc_tau_v_m(u, theta_v_m, tau_v1_m, tau_v2_m):
    """
    Selects time constant for v gate depending on membrane potential.

    Returns `tau_v1_m` below `theta_v_m`, and `tau_v2_m` above.

    Returns
    -------
    float
        Time constant for v gate.
    """
    return tau_v1_m if (u - theta_v_m) < 0 else tau_v2_m

@njit    
def calc_tau_w_m(u, tau_w1_m, tau_w2_m, k_w_m, u_w_m):
    """
    Computes smooth transition time constant for w gate using tanh.

    Returns
    -------
    float
        Blended time constant for w gate.
    """
    return tau_w1_m + (tau_w2_m - tau_w1_m)*(1 + np.tanh(k_w_m*(u - u_w_m)))/2

@njit
def calc_tau_so(u, tau_so1, tau_so2, k_so, u_so):
    """
    Computes tau_so using a sigmoidal transition between two values.
    """
    return tau_so1 + (tau_so2 - tau_so1)*(1 + np.tanh(k_so*(u - u_so)))/2

@njit
def calc_tau_s(u, tau_s1, tau_s2, theta_w):
    """
    Selects tau_s based on threshold.
    """
    return tau_s1 if (u - theta_w) < 0 else tau_s2

@njit
def calc_tau_o(u, tau_o1, tau_o2, theta_o):
    """
    Selects tau_o based on threshold.
    """
    return tau_o1 if (u - theta_o) < 0 else tau_o2
    
@njit
def calc_v_inf(u, theta_v_m):
    """
    Computes the value of v based on membrane potential.
    """
    return 1.0 if u < theta_v_m else 0.0

@njit
def calc_w_inf(u, theta_o, tau_w_inf, w_inf_):
    """
    Computes the value of w based on membrane potential.
    """
    return 1 - u/tau_w_inf if (u - theta_o) < 0 else w_inf_


@njit(parallel=True)
def ionic_kernel_2d(u_new, u, v, w, s, indexes, dt, 
                    u_o, u_u, theta_v, theta_w, theta_v_m,
                    theta_o, tau_v1_m, tau_v2_m, tau_v_p,
                    tau_w1_m, tau_w2_m, k_w_m, u_w_m,
                    tau_w_p, tau_fi, tau_o1, tau_o2,
                    tau_so1, tau_so2, k_so, u_so,
                    tau_s1, tau_s2, k_s, u_s,
                    tau_si, tau_w_inf, w_inf_):
    """
    Computes the ionic kernel for the Bueno-Orovio 2D model.

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

        v[i, j] = calc_v(v[i, j], u[i, j], dt, theta_v, calc_v_inf(u[i, j], theta_v_m), 
                         calc_tau_v_m(u[i, j], theta_v_m, tau_v1_m, tau_v2_m), tau_v_p)
        
        w[i, j] = calc_w(w[i, j], u[i, j], dt, theta_w, calc_w_inf(u[i, j], theta_o, tau_w_inf, w_inf_), 
                         calc_tau_w_m(u[i, j], tau_w1_m, tau_w2_m, k_w_m, u_w_m), tau_w_p)
        
        s[i, j] = calc_s(s[i, j], u[i, j], dt,
                        calc_tau_s(u[i, j], tau_s1, tau_s2, theta_w), k_s, u_s)
        
        J_fi = calc_Jfi(u[i, j], v[i, j], theta_v, u_u, tau_fi)
        J_so = calc_Jso(u[i, j], u_o, theta_w,
                        calc_tau_o(u[i, j], tau_o1, tau_o2, theta_o), 
                        calc_tau_so(u[i, j], tau_so1, tau_so2, k_so, u_so))
        J_si = calc_Jsi(u[i, j], w[i, j], s[i, j], theta_w, tau_si)

        u_new[i, j] += dt * (-J_fi - J_so - J_si)