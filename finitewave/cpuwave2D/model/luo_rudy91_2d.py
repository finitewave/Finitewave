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
    Implements the 2D Luo-Rudy 1991 cardiac model.

    Attributes
    ----------
    state_vars : list
        List of state variable names.
    D_model : float
        Model specific diffusion coefficient.
    """

    def __init__(self):
        """
        Initializes the LuoRudy912D instance, setting up the state variables and parameters.
        """
        CardiacModel.__init__(self)
        self.D_model = 0.1
        self.m = np.ndarray
        self.h = np.ndarray
        self.j_ = np.ndarray
        self.d = np.ndarray
        self.f = np.ndarray
        self.x = np.ndarray
        self.Cai_c = np.ndarray
        self.state_vars = ["u", "m", "h", "j_", "d", "f", "x", "Cai_c"]

    def initialize(self):
        """
        Initializes the state variables.

        This method sets the initial values for the membrane potential ``u``,
        gating variables ``m``, ``h``, ``j_``, ``d``, ``f``, ``x``,
        and intracellular calcium concentration ``Cai_c``.
        """
        super().initialize()
        shape = self.cardiac_tissue.mesh.shape

        self.u = -84.5 * np.ones(shape, dtype=self.npfloat)
        self.u_new = self.u.copy()
        self.m = 0.0017 * np.ones(shape, dtype=self.npfloat)
        self.h = 0.9832 * np.ones(shape, dtype=self.npfloat)
        self.j_ = 0.995484 * np.ones(shape, dtype=self.npfloat)
        self.d = 0.000003 * np.ones(shape, dtype=self.npfloat)
        self.f = np.ones(shape, dtype=self.npfloat)
        self.x = 0.0057 * np.ones(shape, dtype=self.npfloat)
        self.Cai_c = 0.0002 * np.ones(shape, dtype=self.npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel to update the state variables and membrane
        potential.
        """
        ionic_kernel_2d(self.u_new, self.u, self.m, self.h, self.j_, self.d,
                        self.f, self.x, self.Cai_c,
                        self.cardiac_tissue.myo_indexes, self.dt)

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


@njit(parallel=True)
def ionic_kernel_2d(u_new, u, m, h, j_, d, f, x, Cai_c, indexes, dt):
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
    Cai_c : np.ndarray
        Array for the intracellular calcium concentration.
    indexes : np.ndarray
        Array of indexes where the kernel should be computed (``mesh == 1``).
    dt : float
        Time step for the simulation.
    """
    Ko_c = 5.4
    Ki_c = 145
    Nai_c = 18
    Nao_c = 140
    Cao_c = 1.8

    R = 8.314
    T = 310  # Temperature in Kelvin (37°C)
    F = 96.5

    PR_NaK = 0.01833
    E_Na = (R*T/F)*np.log(Nao_c/Nai_c)

    n_j = u.shape[1]

    for ind in prange(len(indexes)):
        ii = indexes[ind]
        i = int(ii / n_j)
        j = ii % n_j

        I_Na = 23 * pow(m[i, j], 3) * h[i, j] * j_[i, j] * (u[i, j] - E_Na)

        alpha_h, beta_h, beta_J, alpha_J = 0, 0, 0, 0
        if u[i, j] >= -40.:
            beta_h = 1. / (0.13 * (1 + np.exp((u[i, j] + 10.66) / -11.1)))
            beta_J = 0.3 * np.exp(-2.535 * 1e-07 *
                                  u[i, j]) / (1 + np.exp(-0.1 * (u[i, j] + 32)))
        else:
            alpha_h = 0.135 * np.exp((80 + u[i, j]) / -6.8)
            beta_h = 3.56 * \
                np.exp(0.079 * u[i, j]) + 3.1 * 1e5 * np.exp(0.35 * u[i, j])
            beta_J = 0.1212 * \
                np.exp(-0.01052 * u[i, j]) / \
                (1 + np.exp(-0.1378 * (u[i, j] + 40.14)))
            alpha_J = (-1.2714 * 1e5 * np.exp(0.2444 * u[i, j]) - 3.474 * 1e-5 * np.exp(-0.04391 * u[i, j])) * \
                      (u[i, j] + 37.78) / (1 + np.exp(0.311 * (u[i, j] + 79.23)))

        alpha_m = 0.32 * (u[i, j] + 47.13) / \
            (1 - np.exp(-0.1 * (u[i, j] + 47.13)))
        beta_m = 0.08 * np.exp(-u[i, j] / 11)

        tau_m = 1. / (alpha_m + beta_m)
        inf_m = alpha_m / (alpha_m + beta_m)
        m[i, j] += dt * (inf_m - m[i, j]) / tau_m

        tau_h = 1. / (alpha_h + beta_h)
        inf_h = alpha_h / (alpha_h + beta_h)
        h[i, j] += dt * (inf_h - h[i, j]) / tau_h

        tau_J = 1. / (alpha_J + beta_J)
        inf_J = alpha_J / (alpha_J + beta_J)
        j_[i, j] += dt * (inf_J - j_[i, j]) / tau_J

        # Slow inward current:
        E_Si = 7.7 - 13.0287 * np.log(Cai_c[i, j])
        I_Si = 0.045 * d[i, j] * f[i, j] * (u[i, j] - E_Si)
        alpha_d = 0.095 * \
            np.exp(-0.01 * (u[i, j] - 5)) / \
            (1 + np.exp(-0.072 * (u[i, j] - 5)))
        beta_d = 0.07 * \
            np.exp(-0.017 * (u[i, j] + 44)) / \
            (1 + np.exp(0.05 * (u[i, j] + 44)))
        alpha_f = 0.012 * \
            np.exp(-0.008 * (u[i, j] + 28)) / \
            (1 + np.exp(0.15 * (u[i, j] + 28)))
        beta_f = 0.0065 * \
            np.exp(-0.02 * (u[i, j] + 30)) / \
            (1 + np.exp(-0.2 * (u[i, j] + 30)))
        Cai_c[i, j] += dt * (-0.0001 * I_Si + 0.07 * (0.0001 - Cai_c[i, j]))

        tau_d = 1. / (alpha_d + beta_d)
        inf_d = alpha_d / (alpha_d + beta_d)
        d[i, j] += dt * (inf_d - d[i, j]) / tau_d

        tau_f = 1. / (alpha_f + beta_f)
        inf_f = alpha_f / (alpha_f + beta_f)
        f[i, j] += dt * (inf_f - f[i, j]) / tau_f

        # Time-dependent potassium current
        E_K = (R * T / F) * \
            np.log((Ko_c + PR_NaK * Nao_c) / (Ki_c + PR_NaK * Nai_c))

        G_K = 0.705 * np.sqrt(Ko_c / 5.4)

        Xi = 0
        if u[i, j] > -100:
            Xi = 2.837 * (np.exp(0.04 * (u[i, j] + 77)) - 1) / \
                ((u[i, j] + 77) * np.exp(0.04 * (u[i, j] + 35)))
        else:
            Xi = 1

        I_K = G_K * x[i, j] * Xi * (u[i, j] - E_K)

        alpha_x = 0.0005 * \
            np.exp(0.083 * (u[i, j] + 50)) / \
            (1 + np.exp(0.057 * (u[i, j] + 50)))
        beta_x = 0.0013 * \
            np.exp(-0.06 * (u[i, j] + 20)) / \
            (1 + np.exp(-0.04 * (u[i, j] + 20)))

        tau_x = 1. / (alpha_x + beta_x)
        inf_x = alpha_x / (alpha_x + beta_x)
        x[i, j] += dt * (inf_x - x[i, j]) / tau_x

        # Time-independent potassium current:
        E_K1 = (R * T / F) * np.log(Ko_c / Ki_c)

        alpha_K1 = 1.02 / (1 + np.exp(0.2385 * (u[i, j] - E_K1 - 59.215)))
        beta_K1 = (0.49124 * np.exp(0.08032 * (u[i, j] - E_K1 + 5.476)) + np.exp(0.06175 * (u[i, j] - E_K1 - 594.31))) / \
                  (1 + np.exp(-0.5143 * (u[i, j] - E_K1 + 4.753)))

        K_1x = alpha_K1 / (alpha_K1 + beta_K1)

        G_K1 = 0.6047 * np.sqrt(Ko_c / 5.4)
        I_K1 = G_K1 * K_1x * (u[i, j] - E_K1)

        # Plateau potassium current:
        E_Kp = E_K1
        K_p = 1. / (1 + np.exp((7.488 - u[i, j]) / 5.98))
        I_Kp = 0.0183 * K_p * (u[i, j] - E_Kp)

        # Background current:
        I_b = 0.03921 * (u[i, j] + 59.87)

        # Total time-independent potassium current:
        I_K1_T = I_K1 + I_Kp + I_b

        u_new[i, j] -= dt * (I_Na + I_Si + I_K1_T + I_K)
