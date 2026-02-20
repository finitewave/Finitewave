import numpy as np
from numba import njit, prange

from .cardiac_model import CardiacModel


class FentonKarma(CardiacModel):
    def __init__(self):
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

        # initial conditions
        self.init_u = 0.0
        self.init_v = 1.0
        self.init_w = 1.0

    def run(self, dt):
        """
        Executes the ionic kernel for the Fenton-Karma model.
        """
        self.counter += 1
        if (self.counter - 1) % self.step != 0:
            return

        ionic_kernel(self.u, self.rhs, self.myo_indexes, dt,
                     self.v, self.w, self.tau_d, self.tau_o, self.tau_r, self.tau_si, 
                        self.tau_v_m, self.tau_v_p, self.tau_w_m, self.tau_w_p,
                        self.k, self.u_c, self.uc_si)
    
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

@njit(parallel=True, fastmath=True, cache=True)
def ionic_kernel(u, rhs, indexes, dt,
                 v, w, tau_d, tau_o, tau_r, tau_si, 
                 tau_v_m, tau_v_p, tau_w_m, tau_w_p, k, u_c, uc_si):
    """
    Computes the ionic kernel for the Fenton-Karma 2D model.

    Parameters
    ----------
    u : np.ndarray
        Current action potential array.
    rhs : np.ndarray
        Array to store the updated action potential values.
    indexes : np.ndarray
        Array of indices where the kernel should be computed (``mesh == 1``).
    dt : float
        Time step for the simulation.
    v : np.ndarray
        Fast recovery variable array.
    w : np.ndarray
        Slow recovery variable array.
    tau_d : float
        Time constant for depolarization.
    tau_o : float
        Time constant for subthreshold repolarization.
    tau_r : float
        Time constant for suprathreshold repolarization.
    tau_si : float
        Time constant for the slow inward current.
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
    """

    for i in prange(len(indexes)):
        ii = indexes[i]

        v.flat[ii] = calc_v(v.flat[ii], u.flat[ii], dt, u_c, tau_v_m, tau_v_p)
        w.flat[ii] = calc_w(w.flat[ii], u.flat[ii], dt, u_c, tau_w_m, tau_w_p)

        J_fi = calc_Jfi(u.flat[ii], v.flat[ii], u_c, tau_d)
        J_so = calc_Jso(u.flat[ii], u_c, tau_o, tau_r)
        J_si = calc_Jsi(u.flat[ii], v.flat[ii], k, uc_si, tau_si)

        rhs.flat[ii] = (-J_fi - J_so - J_si)

