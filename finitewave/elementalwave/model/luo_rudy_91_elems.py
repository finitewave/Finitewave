import numpy as np
import math
from numba import njit, prange

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.elementalwave.stencil.triangle_isotropic_stencil import (
    TriangleIsotropicStencil
)


class LuoRudy91Elems(CardiacModel):
    def __init__(self):
        super().__init__()
        self.cardiac_tissue = None
        self.stencil = None
        self.stim_sequence = None
        self._rhs = np.ndarray
        self._u = np.ndarray
        self._u_new = np.ndarray
        self.D_model = 0.1

        self.m = np.ndarray
        self.h = np.ndarray
        self.j = np.ndarray
        self.d = np.ndarray
        self.f = np.ndarray
        self.x = np.ndarray
        self.cai = np.ndarray

        self.state_vars = ["u", "m", "h", "j", "d", "f", "x", "cai"]
        self.npfloat = 'float64'

        # Ion Channel Conductances (mS/µF)
        self.gna = 23.0     # Fast sodium (Na+) conductance
        self.gsi = 0.045     # Slow inward calcium (Ca2+) conductance
        self.gk = 0.705    # Time-dependent potassium (K+) conductance
        self.gk1 = 0.6047   # Inward rectifier potassium (K1) conductance
        self.gkp = 0.0183   # Plateau potassium (Kp) conductance
        self.gb = 0.03921  # Background conductance (leak current)

        # Extracellular and Intracellular Ion Concentrations (mM)
        self.ko = 5.4      # Extracellular potassium concentration
        self.ki = 145.0    # Intracellular potassium concentration
        self.nai = 18.0     # Intracellular sodium concentration
        self.nao = 140.0    # Extracellular sodium concentration
        self.cao = 1.8      # Extracellular calcium concentration

        # Physical Constants
        self.R = 8.314      # Universal gas constant (J/(mol·K))
        self.T = 310.0      # Temperature (Kelvin, 37°C)
        self.F = 96.5       # Faraday constant (C/mmol)

        # Ion Permeability Ratios
        self.PR_NaK = 0.01833  # Na+/K+ permeability ratio
        self.E_Na = 54.7857
        self.E_K1 = -87.8789

        # initial conditions
        self.init_u = -84.5
        self.init_m = 0.0017
        self.init_h = 0.9832
        self.init_j = 0.995484
        self.init_d = 0.000003
        self.init_f = 1.0
        self.init_x = 0.0057
        self.init_cai = 0.0002

    def initialize(self):
        super().initialize()
        self.u[:] = self.init_u * np.ones_like(self.u)
        self._u = self.u[self.cardiac_tissue.myo_indexes].copy()
        self._rhs = np.zeros_like(self._u)
        self._u_new = self._u.copy()
        self.v = np.zeros_like(self.u)
        self.u_new = self.u

        self.m = self.init_m * np.ones_like(self.u)
        self.h = self.init_h * np.ones_like(self.u)
        self.j = self.init_j * np.ones_like(self.u)
        self.d = self.init_d * np.ones_like(self.u)
        self.f = self.init_f * np.ones_like(self.u)
        self.x = self.init_x * np.ones_like(self.u)
        self.cai = self.init_cai * np.ones_like(self.u)

    def run(self, initialize=True, num_of_threads=None):
        super().run(initialize, num_of_threads)
        self.u[self.cardiac_tissue.myo_indexes] = self._u

    def swap_arrays(self):
        self._u_new, self._u = self._u, self._u_new

    def run_diffusion_kernel(self):
        """
        Executes the diffusion kernel computation using the current parameters
        and tissue weights.
        """
        self._u_new = self.diffusion_kernel(self._u_new, self._u, self._rhs,
                                            self.weights)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel for the Aliev-Panfilov model.
        """
        ionic_kernel_elems(self._u, self.m, self.h, self.j, self.d, self.f,
                           self.x, self.cai,
                           self._rhs, self.cardiac_tissue.myo_indexes, self.dt,
                           self.gna, self.gsi, self.gk, self.gk1, self.gkp,
                           self.gb, self.ko, self.ki, self.nai, self.nao,
                           self.cao, self.R, self.T, self.F, self.PR_NaK,
                           self.E_Na, self.E_K1)

    def select_stencil(self, cardiac_tissue):
        if cardiac_tissue.elems.shape[1] == 3:
            return TriangleIsotropicStencil()
        raise ValueError


@njit(parallel=True)
def ionic_kernel_elems(u, m, h, j_, d, f, x, cai, rhs, indexes, dt, gna, gsi,
                       gk, gk1, gkp, gb, ko, ki, nai, nao, cao, R, T, F,
                       PR_NaK, E_Na, E_K1):
    """
    Computes the ionic currents and updates the state variables in the 2D
    Luo-Rudy 1991 cardiac model.

    Parameters
    ----------
    u_new : np.ndarray
        Array to store the updated membrane potential.
    u : np.ndarray
        Array of the current membrane potential values.
    m : np.ndarray
        Array for the gating variable `m`.
    h : np.ndarray
        Array for the gating variable `h`.
    j_ : np.ndarray
        Array for the gating variable `j_`.
    d : np.ndarray
        Array for the gating variable `d`.
    f : np.ndarray
        Array for the gating variable `f`.
    x : np.ndarray
        Array for the gating variable `x`.
    cai : np.ndarray
        Array for the intracellular calcium concentration.
    indexes : np.ndarray
        Array of indexes where the kernel should be computed (``mesh == 1``).
    dt : float
        Time step for the simulation.
    """

    for i in prange(len(indexes)):
        ind = indexes[i]

        # Fast sodium current:
        m[ind] += dt*calc_dm(u[i], m[ind])
        h[ind] += dt*calc_dh(u[i], h[ind])
        j_[ind] += dt*calc_dj(u[i], j_[ind])

        ina = calc_ina(u[i], m[ind], h[ind], j_[ind], E_Na, gna)

        # Slow inward current:
        d[ind] += dt*calc_dd(u[i], d[ind])
        f[ind] += dt*calc_df(u[i], f[ind])

        isi = calc_isk(u[i], d[ind], f[ind], cai[ind], gsi)

        cai[ind] += dt*calc_dcai(cai[ind], isi)

        # Time-dependent potassium current:
        x[ind] += dt*calc_dx(u[i], x[ind])

        ik = calc_ik(u[i], x[ind], ko, ki, nao, nai, PR_NaK, R, T, F, gk)

        # Time-independent potassium current:
        ik1 = calc_ik1(u[i], ko, E_K1, gk1)

        # Plateau potassium current:
        ikp = calc_ikp(u[i], E_K1, gkp)

        # Background current:
        ib = calc_ib(u[i], gb)

        rhs[i] = dt*calc_rhs(ina, isi, ik, ik1, ikp, ib)


@njit
def calc_rhs(ina, isi, ik, ik1, ikp, ib):
    """
    Computes the right-hand side of the Luo-Rudy-91 model for the transmembrane potential u.
    This function implements the ordinary differential equation governing the
    evolution of the transmembrane potential `u`, which represents the electrical
    activity of cardiac cells. The equation includes contributions from various ionic
    currents that influence the membrane potential.

    Parameters
    ----------
    ina : float
        Fast sodium current [μA/μF].
    isi : float
        Slow inward calcium current [μA/μF].
    ik : float 
        Time-dependent potassium current [μA/μF].
    ik1 : float
        Time-independent inward rectifier potassium current [μA/μF].
    ikp : float
        Plateau potassium current [μA/μF].
    ib : float
        Background (leak) current [μA/μF].

    Returns
    -------
    float
        The computed right-hand side of the differential equation for `u`.
    """
    # Total time-independent potassium current:
    ik1t = ik1 + ikp + ib

    return -(ina + isi + ik1t + ik)


@njit
def calc_dm(u, m, exp=math.exp):
    """
    Computes the gating variable m for the fast sodium current (I_Na).
    Gating variable m follows Hodgkin-Huxley kinetics with voltage-dependent
    time constants and steady-state values.

    Parameters
    ----------
    u : float
        Membrane potential [mV].
    m : float
        Current value of the gating variable m.
    exp : callable, optional
        Exponential function to use (default: math.exp).

    Returns
    -------
    dm_dt : float
        Time derivative of the gating variable m.
    """
    alpha_m = 0.32 * (u + 47.13)/(1 - exp(-0.1 * (u + 47.13)))
    beta_m = 0.08 * exp(-u / 11)

    tau_m = 1. / (alpha_m + beta_m)
    inf_m = alpha_m / (alpha_m + beta_m)

    return (inf_m - m) / tau_m


@njit
def calc_dh(u, h, exp=math.exp):
    """
    Computes the gating variable h for the fast sodium current (I_Na).
    Gating variable h follows Hodgkin-Huxley kinetics with voltage-dependent
    time constants and steady-state values.

    Parameters
    ----------
    u : float
        Membrane potential [mV].
    h : float
        Current value of the gating variable h.
    exp : callable, optional
        Exponential function to use (default: math.exp).

    Returns
    -------
    dh_dt : float
        Time derivative of the gating variable h.
    """
    alpha_h, beta_h = 0, 0
    if u >= -40.:
        beta_h = 1. / (0.13 * (1 + exp((u + 10.66) / -11.1)))
    else:
        alpha_h = 0.135 * exp((80 + u) / -6.8)
        beta_h = 3.56 * exp(0.079 * u) + 3.1 * 1e5 * exp(0.35 * u)

    tau_h = 1. / (alpha_h + beta_h)
    inf_h = alpha_h / (alpha_h + beta_h)

    return (inf_h - h) / tau_h


@njit
def calc_dj(u, j, exp=math.exp):
    """
    Computes the gating variable j for the fast sodium current (I_Na).
    Gating variable j follows Hodgkin-Huxley kinetics with voltage-dependent
    time constants and steady-state values.
    
    Parameters
    ----------
    u : float
        Membrane potential [mV].
    j : float
        Current value of the gating variable j.
    exp : callable, optional
        Exponential function to use (default: math.exp).

    Returns
    -------
    dj_dt : float
        Time derivative of the gating variable j.
    """

    alpha_j, beta_j = 0, 0
    if u >= -40.:
        beta_j = 0.3 * exp(-2.535 * 1e-07 * u) / (1 + exp(-0.1 * (u + 32)))
    else:
        beta_j = 0.1212 * exp(-0.01052 * u) / (1 + exp(-0.1378 * (u + 40.14)))
        alpha_j = (-1.2714 * 1e5 * exp(0.2444 * u) - 3.474 * 1e-5 * exp(-0.04391 * u)) * \
                    (u + 37.78) / (1 + exp(0.311 * (u + 79.23)))

    tau_j = 1. / (alpha_j + beta_j)
    inf_j = alpha_j / (alpha_j + beta_j)

    return (inf_j - j) / tau_j


@njit
def calc_dd(u, d, exp=math.exp):
    """
    Computes the gating variable d for the slow inward calcium current (I_Si).
    Gating variable d follows Hodgkin-Huxley kinetics with voltage-dependent
    time constants and steady-state values.

    Parameters
    ----------
    u : float
        Membrane potential [mV].
    d : float
        Current value of the gating variable d.
    exp : callable, optional
        Exponential function to use (default: math.exp).

    Returns
    -------
    dd_dt : float
        Time derivative of the gating variable d.
    """
    alpha_d = 0.095 * exp(-0.01 * (u - 5)) / (1 + exp(-0.072 * (u - 5)))
    beta_d = 0.07 * exp(-0.017 * (u + 44)) / (1 + exp(0.05 * (u + 44)))

    tau_d = 1. / (alpha_d + beta_d)
    inf_d = alpha_d / (alpha_d + beta_d)

    return (inf_d - d) / tau_d


@njit
def calc_df(u, f, exp=math.exp):
    """
    Computes the gating variable f for the slow inward calcium current (I_Si).
    Gating variable f follows Hodgkin-Huxley kinetics with voltage-dependent
    time constants and steady-state values.

    Parameters
    ----------
    u : float
        Membrane potential [mV].
    f : float
        Current value of the gating variable f.
    exp : callable, optional
        Exponential function to use (default: math.exp).

    Returns
    -------
    df_dt : float
        Time derivative of the gating variable f.
    """
    alpha_f = 0.012 * exp(-0.008 * (u + 28)) / (1 + exp(0.15 * (u + 28)))
    beta_f = 0.0065 * exp(-0.02 * (u + 30)) / (1 + exp(-0.2 * (u + 30)))

    tau_f = 1. / (alpha_f + beta_f)
    inf_f = alpha_f / (alpha_f + beta_f)

    return (inf_f - f) / tau_f

@njit
def calc_dx(u, x, exp=math.exp):
    """
    Computes the gating variable x for the time-dependent potassium current (I_K).
    Gating variable x follows Hodgkin-Huxley kinetics with voltage-dependent
    time constants and steady-state values.

    Parameters
    ----------
    u : float
        Membrane potential [mV].
    x : float
        Current value of the gating variable x.
    exp : callable, optional
        Exponential function to use (default: math.exp).

    Returns
    -------
    dx_dt : float
        Time derivative of the gating variable x.
    """
    alpha_x = 0.0005 * exp(0.083 * (u + 50)) / (1 + exp(0.057 * (u + 50)))
    beta_x = 0.0013 * exp(-0.06 * (u + 20)) / (1 + exp(-0.04 * (u + 20)))

    tau_x = 1. / (alpha_x + beta_x)
    inf_x = alpha_x / (alpha_x + beta_x)

    return (inf_x - x) / tau_x


@njit
def calc_dcai(cai, isi):
    """
    Computes the time derivative of intracellular calcium concentration (cai).

    Intracellular calcium dynamics are influenced by the slow inward calcium current (I_Si).
    A portion of I_Si contributes to changes in cai, while a constant leak term restores
    cai toward a baseline level.

    Parameters
    ----------
    cai : float
        Current intracellular calcium concentration [mM].
    isi : float
        Slow inward calcium current [μA/μF].

    Returns
    -------
    dcai_dt : float
        Time derivative of intracellular calcium concentration.
    """
    return -0.0001 * isi + 0.07 * (0.0001 - cai)


@njit
def calc_ina(u, m, h, j, E_Na, gna):
    """
    Computes the fast inward sodium current (I_Na).

    I_Na is responsible for the rapid depolarization (phase 0) of the action potential. 
    It depends on three gates:
    - m: activation gate (opens quickly),
    - h: fast inactivation gate,
    - j: slow inactivation gate.

    Gating dynamics follow Hodgkin-Huxley kinetics with voltage-dependent time constants 
    and steady-state values. I_Na = g_Na * m^3 * h * j * (u - E_Na).

    Parameters
    ----------
    u : float
        Membrane potential [mV].
    m, h, j : float
        Gating variables for the sodium channel.
    E_Na : float
        Reversal potential for Na⁺ [mV].
    gna : float
        Maximal sodium conductance [mS/μF].

    Returns
    -------
    ina : float
        Fast sodium current [μA/μF].
    """

    return gna * m * m * m * h * j * (u - E_Na)


@njit
def calc_isk(u, d, f, cai, gsi, log=math.log):
    """
    Computes the slow inward calcium current (I_Si).

    I_Si is primarily carried by L-type Ca²⁺ channels and governs the plateau (phase 2).
    The reversal potential E_Si is dynamically calculated based on intracellular Ca²⁺ levels.

    Calcium handling is simplified: part of I_Si is subtracted from intracellular Ca²⁺, 
    while a constant leak term restores it toward a baseline.

    Parameters
    ----------
    u : float
        Membrane potential [mV].
    d, f : float
        Activation and inactivation gates of the calcium channel.
    cai : float
        Intracellular calcium concentration [mM].
    gsi : float
        Maximal calcium conductance [mS/μF].
    log : callable, optional
        Logarithm function to use (default: math.log).

    Returns
    -------
    I_Si : float
        Slow inward calcium current [μA/μF].
    """
    E_Si = 7.7 - 13.0287 * log(cai)

    return gsi * d * f * (u - E_Si)


@njit
def calc_ik(u, x, ko, ki, nao, nai, PR_NaK, R, T, F, gk, sqrt=math.sqrt, exp=math.exp, log=math.log):
    """
    Computes the time-dependent outward potassium current (I_K).

    This current drives late repolarization (phase 3) and is voltage- and time-dependent.
    Reversal potential is calculated via the Goldman-Hodgkin-Katz equation 
    (with sodium/potassium permeability ratio).

    An auxiliary factor Xi introduces voltage-sensitive activation near -100 mV.

    Parameters
    ----------
    u : float
        Membrane potential [mV].
    x : float
        Activation gate of the delayed rectifier K⁺ channel.
    ko, ki : float
        Extra-/intracellular potassium concentrations [mM].
    nao, nai : float
        Extra-/intracellular sodium concentrations [mM].
    PR_NaK : float
        Na⁺/K⁺ permeability ratio.
    R, T, F : float
        Gas constant, temperature [K], and Faraday constant.
    gk : float
        Maximum potassium conductance [mS/μF].
    sqrt, exp, log : callable, optional
        Mathematical functions to use (default: math.sqrt, math.exp, math.log).

    Returns
    -------
    I_K : float
        Time-dependent potassium current [μA/μF].
    """
    E_K = (R * T / F) * \
        log((ko + PR_NaK * nao) / (ki + PR_NaK * nai))

    G_K = gk * sqrt(ko / 5.4)

    Xi = 0
    if u > -100:
        Xi = 2.837 * (exp(0.04 * (u + 77)) - 1) / \
            ((u + 77) * exp(0.04 * (u + 35)))
    else:
        Xi = 1

    return G_K * x * Xi * (u - E_K)


@njit
def calc_ik1(u, ko, E_K1, gk1, exp=math.exp, sqrt=math.sqrt):
    """
    Computes the time-independent inward rectifier potassium current (I_K1).

    I_K1 stabilizes the resting membrane potential and contributes to 
    late repolarization. It is primarily active at negative voltages and 
    follows a voltage-dependent gating-like term (K1_x).

    Parameters
    ----------
    u : float
        Membrane potential [mV].
    ko : float
        Extracellular potassium [mM].
    E_K1 : float
        Equilibrium potential for K1 current [mV].
    gk1 : float
        Maximum K1 conductance [mS/μF].
    exp, sqrt : callable, optional
        Mathematical functions to use (default: math.exp, math.sqrt).

    Returns
    -------
    I_K1 : float
        Time-independent K⁺ current [μA/μF].
    """
    alpha_K1 = 1.02 / (1 + exp(0.2385 * (u - E_K1 - 59.215)))
    beta_K1 = (0.49124 * exp(0.08032 * (u - E_K1 + 5.476)) + exp(0.06175 * (u - E_K1 - 594.31))) / \
                (1 + exp(-0.5143 * (u - E_K1 + 4.753)))

    K_1x = alpha_K1 / (alpha_K1 + beta_K1)

    G_K1 = gk1 * sqrt(ko / 5.4)

    return G_K1 * K_1x * (u - E_K1)


@njit
def calc_ikp(u, E_K1, gkp, exp=math.exp):
    """
    Computes the plateau potassium current (I_Kp).

    I_Kp is a small, quasi-steady outward current that operates in the 
    plateau phase. Its activation is a sigmoid function of voltage.

    Parameters
    ----------
    u : float
        Membrane potential [mV].
    ko : float
        Extracellular potassium [mM].
    E_K1 : float
        Equilibrium potential (same as I_K1).
    gkp : float
        Plateau potassium conductance [mS/μF].
    exp : callable, optional
        Exponential function to use (default: math.exp).

    Returns
    -------
    I_Kp : float
        Plateau potassium current [μA/μF].
    """
    E_Kp = E_K1
    K_p = 1. / (1 + exp((7.488 - u) / 5.98))

    return gkp * K_p * (u - E_Kp)


@njit
def calc_ib(u, gb):
    """
    Computes the non-specific background (leak) current.

    This is a linear leak current contributing to resting potential maintenance.

    Parameters
    ----------
    u : float
        Membrane potential [mV].
    gb : float
        Background conductance [mS/μF].

    Returns
    -------
    I_b : float
        Background current [μA/μF].
    """
    return gb * (u + 59.87)
