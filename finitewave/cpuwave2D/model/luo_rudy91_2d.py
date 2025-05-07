import numpy as np
from numba import njit, prange
from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import (
    AsymmetricStencil2D
)
from finitewave.cpuwave2D.stencil.isotropic_stencil_2d import (
    IsotropicStencil2D
)


class LuoRudy912D(CardiacModel):
    """
    Implements the Luo-Rudy 1991 ventricular action potential model.

    This biophysically detailed model simulates the ionic currents and membrane potential 
    of a ventricular cardiac cell based on Hodgkin-Huxley-type formalism. It was one of 
    the first to incorporate realistic ionic channel kinetics, calcium dynamics, and 
    multiple potassium currents to reproduce key phases of the action potential.

    The model includes:
    - Fast Na⁺ current (I_Na)
    - Slow inward Ca²⁺ current (I_Si)
    - Time-dependent K⁺ current (I_K)
    - Time-independent K⁺ current (I_K1)
    - Plateau K⁺ current (I_Kp)
    - Background/leak current (I_b)

    Attributes
    ----------
    state_vars : list of str
        List of state variable names to save and restore (`u`, `m`, `h`, `j`, `d`, `f`, `x`, `cai`).
    D_model : float
        Diffusion coefficient representing electrical conductivity in the medium (typically set to 0.1).
    gna, gsi, gk, gk1, gkp, gb : float
        Maximum conductances for Na⁺, Ca²⁺, K⁺, and background channels [mS/μF].
    ko, ki, nao, nai, cao : float
        Ion concentrations in mM (extracellular and intracellular for Na⁺, K⁺, Ca²⁺).
    R, T, F : float
        Physical constants: gas constant, temperature in Kelvin, and Faraday constant.
    PR_NaK : float
        Sodium/potassium permeability ratio (used in reversal potential calculation for I_K).

    Paper
    -----
    Luo CH, Rudy Y. 
    A model of the ventricular cardiac action potential. Depolarization, repolarization, and their interaction. 
    Circ Res. 1991 Jun;68(6):1501-26. 
    doi: 10.1161/01.res.68.6.1501. 
    PMID: 1709839.

    """

    def __init__(self):
        """
        Initializes the LuoRudy912D instance, setting up the state variables and parameters.
        """
        CardiacModel.__init__(self)
        self.D_model = 0.1

        self.m   = np.ndarray
        self.h   = np.ndarray
        self.j   = np.ndarray
        self.d   = np.ndarray
        self.f   = np.ndarray
        self.x   = np.ndarray
        self.cai = np.ndarray
        
        self.state_vars = ["u", "m", "h", "j", "d", "f", "x", "cai"]
        self.npfloat = 'float64'

        # Ion Channel Conductances (mS/µF)
        self.gna = 23.0     # Fast sodium (Na+) conductance
        self.gsi = 0.09     # Slow inward calcium (Ca2+) conductance
        self.gk  = 0.282    # Time-dependent potassium (K+) conductance
        self.gk1 = 0.6047   # Inward rectifier potassium (K1) conductance
        self.gkp = 0.0183   # Plateau potassium (Kp) conductance
        self.gb  = 0.03921  # Background conductance (leak current)

        # Extracellular and Intracellular Ion Concentrations (mM)
        self.ko  = 5.4      # Extracellular potassium concentration
        self.ki  = 145.0    # Intracellular potassium concentration
        self.nai = 18.0     # Intracellular sodium concentration
        self.nao = 140.0    # Extracellular sodium concentration
        self.cao = 1.8      # Extracellular calcium concentration

        # Physical Constants
        self.R = 8.314      # Universal gas constant (J/(mol·K))
        self.T = 310.0      # Temperature (Kelvin, 37°C)
        self.F = 96.5       # Faraday constant (C/mmol)

        # Ion Permeability Ratios
        self.PR_NaK = 0.01833  # Na+/K+ permeability ratio

    def initialize(self):
        """
        Initializes the state variables.

        This method sets the initial values for the membrane potential ``u``,
        gating variables ``m``, ``h``, ``j``, ``d``, ``f``, ``x``,
        and intracellular calcium concentration ``cai``.
        """
        super().initialize()
        shape = self.cardiac_tissue.mesh.shape

        self.u = -84.5 * np.ones(shape, dtype=self.npfloat)
        self.u_new = self.u.copy()
        self.m = 0.0017 * np.ones(shape, dtype=self.npfloat)
        self.h = 0.9832 * np.ones(shape, dtype=self.npfloat)
        self.j = 0.995484 * np.ones(shape, dtype=self.npfloat)
        self.d = 0.000003 * np.ones(shape, dtype=self.npfloat)
        self.f = np.ones(shape, dtype=self.npfloat)
        self.x = 0.0057 * np.ones(shape, dtype=self.npfloat)
        self.cai = 0.0002 * np.ones(shape, dtype=self.npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel to update the state variables and membrane
        potential.
        """
        ionic_kernel_2d(self.u_new, self.u, 
                        self.m, self.h, self.j, self.d,self.f, self.x, self.cai, 
                        self.cardiac_tissue.myo_indexes, self.dt, 
                        self.gna, self.gsi, self.gk, self.gk1, self.gkp, self.gb, 
                        self.ko, self.ki, self.nai, self.nao, self.cao, self.R, self.T, self.F, self.PR_NaK)

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
def calc_gating_var(var, dt, alpha, beta):
    """
    Computes the gating variable dynamics based on the Hodgkin-Huxley formalism.

    Parameters
    ----------
    var : float
        Current value of the gating variable.
    dt : float
        Time step [ms].
    alpha : float
        Rate constant for activation.
    beta : float
        Rate constant for inactivation.

    Returns
    -------
    var_new : float
        Updated gating variable.
    """
    tau = 1. / (alpha + beta)
    inf = alpha / (alpha + beta)
    var += dt * (inf - var) / tau
    return var

@njit
def calc_ina(u, dt, m, h, j, E_Na, gna):
    """
    Computes the fast inward sodium current (I_Na) and updates gating variables m, h, j.

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
    dt : float
        Time step [ms].
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
    m, h, j : float
        Updated gating variables.
    """
    alpha_h, beta_h, beta_J, alpha_J = 0, 0, 0, 0
    if u >= -40.:
        beta_h = 1. / (0.13 * (1 + np.exp((u + 10.66) / -11.1)))
        beta_J = 0.3 * np.exp(-2.535 * 1e-07 *
                                u) / (1 + np.exp(-0.1 * (u + 32)))
    else:
        alpha_h = 0.135 * np.exp((80 + u) / -6.8)
        beta_h = 3.56 * \
            np.exp(0.079 * u) + 3.1 * 1e5 * np.exp(0.35 * u)
        beta_J = 0.1212 * \
            np.exp(-0.01052 * u) / \
            (1 + np.exp(-0.1378 * (u + 40.14)))
        alpha_J = (-1.2714 * 1e5 * np.exp(0.2444 * u) - 3.474 * 1e-5 * np.exp(-0.04391 * u)) * \
                    (u + 37.78) / (1 + np.exp(0.311 * (u + 79.23)))

    alpha_m = 0.32 * (u + 47.13) / \
        (1 - np.exp(-0.1 * (u + 47.13)))
    beta_m = 0.08 * np.exp(-u / 11)

    m = calc_gating_var(m, dt, alpha_m, beta_m)
    h = calc_gating_var(h, dt, alpha_h, beta_h)
    j = calc_gating_var(j, dt, alpha_J, beta_J)

    return gna * m * m * m * h * j * (u - E_Na), m, h, j

@njit
def calc_isk(u, dt, d, f, cai, gsi):
    """
    Computes the slow inward calcium current (I_Si) and updates d, f, and intracellular calcium.

    I_Si is primarily carried by L-type Ca²⁺ channels and governs the plateau (phase 2).
    The reversal potential E_Si is dynamically calculated based on intracellular Ca²⁺ levels.

    Calcium handling is simplified: part of I_Si is subtracted from intracellular Ca²⁺, 
    while a constant leak term restores it toward a baseline.

    Parameters
    ----------
    u : float
        Membrane potential [mV].
    dt : float
        Time step [ms].
    d, f : float
        Activation and inactivation gates of the calcium channel.
    cai : float
        Intracellular calcium concentration [mM].
    gsi : float
        Maximal calcium conductance [mS/μF].

    Returns
    -------
    I_Si : float
        Slow inward calcium current [μA/μF].
    d, f, cai : float
        Updated gating variables and intracellular Ca²⁺.
    """
    E_Si = 7.7 - 13.0287 * np.log(cai)
    I_Si = gsi * d * f * (u - E_Si)
    alpha_d = 0.095 * \
        np.exp(-0.01 * (u - 5)) / \
        (1 + np.exp(-0.072 * (u - 5)))
    beta_d = 0.07 * \
        np.exp(-0.017 * (u + 44)) / \
        (1 + np.exp(0.05 * (u + 44)))
    alpha_f = 0.012 * \
        np.exp(-0.008 * (u + 28)) / \
        (1 + np.exp(0.15 * (u + 28)))
    beta_f = 0.0065 * \
        np.exp(-0.02 * (u + 30)) / \
        (1 + np.exp(-0.2 * (u + 30)))
    
    d = calc_gating_var(d, dt, alpha_d, beta_d)
    f = calc_gating_var(f, dt, alpha_f, beta_f)

    cai += dt * (-0.0001 * I_Si + 0.07 * (0.0001 - cai))

    return I_Si, d, f, cai

@njit
def calc_ik(u, dt, x, ko, ki, nao, nai, PR_NaK, R, T, F, gk):
    """
    Computes the time-dependent outward potassium current (I_K) and updates gate x.

    This current drives late repolarization (phase 3) and is voltage- and time-dependent.
    Reversal potential is calculated via the Goldman-Hodgkin-Katz equation 
    (with sodium/potassium permeability ratio).

    An auxiliary factor Xi introduces voltage-sensitive activation near -100 mV.

    Parameters
    ----------
    u : float
        Membrane potential [mV].
    dt : float
        Time step [ms].
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

    Returns
    -------
    I_K : float
        Time-dependent potassium current [μA/μF].
    x : float
        Updated activation gate.
    """
    E_K = (R * T / F) * \
        np.log((ko + PR_NaK * nao) / (ki + PR_NaK * nai))

    G_K = gk * np.sqrt(ko / 5.4)

    Xi = 0
    if u > -100:
        Xi = 2.837 * (np.exp(0.04 * (u + 77)) - 1) / \
            ((u + 77) * np.exp(0.04 * (u + 35)))
    else:
        Xi = 1

    I_K = G_K * x * Xi * (u - E_K)

    alpha_x = 0.0005 * \
        np.exp(0.083 * (u + 50)) / \
        (1 + np.exp(0.057 * (u + 50)))
    beta_x = 0.0013 * \
        np.exp(-0.06 * (u + 20)) / \
        (1 + np.exp(-0.04 * (u + 20)))
    
    x = calc_gating_var(x, dt, alpha_x, beta_x)

    return I_K, x

@njit
def calc_ik1(u, ko, E_K1, gk1):
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

    Returns
    -------
    I_K1 : float
        Time-independent K⁺ current [μA/μF].
    """
    alpha_K1 = 1.02 / (1 + np.exp(0.2385 * (u - E_K1 - 59.215)))
    beta_K1 = (0.49124 * np.exp(0.08032 * (u - E_K1 + 5.476)) + np.exp(0.06175 * (u - E_K1 - 594.31))) / \
                (1 + np.exp(-0.5143 * (u - E_K1 + 4.753)))

    K_1x = alpha_K1 / (alpha_K1 + beta_K1)

    G_K1 = gk1 * np.sqrt(ko / 5.4)
    I_K1 = G_K1 * K_1x * (u - E_K1)

    return I_K1

@njit
def calc_ikp(u, E_K1, gkp):
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

    Returns
    -------
    I_Kp : float
        Plateau potassium current [μA/μF].
    """
    E_Kp = E_K1
    K_p = 1. / (1 + np.exp((7.488 - u) / 5.98))
    I_Kp = gkp * K_p * (u - E_Kp)

    return I_Kp

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


@njit(parallel=True)
def ionic_kernel_2d(u_new, u, m, h, j_, d, f, x, cai, indexes, dt, gna, gsi, gk, gk1, gkp, gb, ko, ki, nai, nao, cao, R, T, F, PR_NaK):
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

    E_Na = (R*T/F)*np.log(nao/nai)

    n_j = u.shape[1]

    for ind in prange(len(indexes)):
        ii = indexes[ind]
        i = int(ii / n_j)
        j = ii % n_j

        # Fast sodium current:
        ina, m[i, j], h[i, j], j_[i, j] = calc_ina(u[i, j], dt, m[i, j], h[i, j], j_[i, j], E_Na, gna)

        # Slow inward current:
        isi, d[i, j], f[i, j], cai[i, j] = calc_isk(u[i, j], dt, d[i, j], f[i, j], cai[i, j], gsi)

        # Time-dependent potassium current:
        ik, x[i, j] = calc_ik(u[i, j], dt, x[i, j], ko, ki, nao, nai, PR_NaK, R, T, F, gk)

        E_K1 = (R * T / F) * np.log(ko / ki)

        # Time-independent potassium current:
        ik1 = calc_ik1(u[i, j], ko, E_K1, gk1)

        # Plateau potassium current:
        ikp = calc_ikp(u[i, j], E_K1, gkp)

        # Background current:
        ib = calc_ib(u[i, j], gb)

        # Total time-independent potassium current:
        ik1t = ik1 + ikp + ib

        u_new[i, j] -= dt * (ina + isi + ik1t + ik)




