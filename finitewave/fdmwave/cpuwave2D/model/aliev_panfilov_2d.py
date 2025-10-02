import numpy as np
from numba import njit, prange

from finitewave.core.model.cardiac_model import CardiacModel


class AlievPanfilov2D(CardiacModel):
    """
    Two-dimensional implementation of the Aliev–Panfilov model of cardiac excitation.

    The Aliev–Panfilov model is a phenomenological two-variable model designed to
    reproduce basic features of cardiac excitation, including wave propagation and
    reentry, while remaining computationally efficient. It uses a single recovery
    variable coupled with a cubic nonlinearity to simulate action potential dynamics
    in excitable media.

    Attributes
    ----------
    u : np.ndarray
        Transmembrane potential (dimensionless, normalized to [0,1]).
    v : np.ndarray
        Recovery variable describing refractoriness.
    D_model : float
        Diffusion coefficient used for simulating spatial propagation.
    state_vars : list of str
        Names of the state variables to be saved and restored.
    npfloat : str
        Floating-point precision used in the simulation (default: 'float64').

    Model Parameters
    ----------------
    a : float
        Excitability threshold parameter.
    k : float
        Strength of the nonlinear source term (governs spike shape).
    eap : float
        Baseline recovery rate.
    mu_1 : float
        Recovery rate coefficient (scales v feedback).
    mu_2 : float
        Recovery rate offset (modulates u-dependence of recovery).

    Paper
    -----
    Rubin R. Aliev, Alexander V. Panfilov,
    A simple two-variable model of cardiac excitation,
    Chaos, Solitons & Fractals,
    Volume 7, Issue 3,
    1996,
    Pages 293-301,
    ISSN 0960-0779,
    https://doi.org/10.1016/0960-0779(95)00089-5.

    Attributes
    ----------
    v : np.ndarray
        Array for the recovery variable.
    w : np.ndarray
        Array for diffusion weights.
    D_model : float
        Model specific diffusion coefficient
    state_vars : list
        List of state variables to be saved and restored.
    npfloat : str
        Data type used for floating-point operations, default is 'float64'.
    """

    def __init__(self):
        """
        Initializes the AlievPanfilov2D instance with default parameters.
        """
        super().__init__()
        self.v = np.ndarray

        self.state_vars = ["u", "v"]
        self.npfloat = 'float64'
        self.D_model = 1.

        # model parameters
        self.a = 0.1
        self.k = 8.0
        self.eap = 0.01
        self.mu_1 = 0.2
        self.mu_2 = 0.3

        # initial conditions
        self.init_u = 0.0
        self.init_v = 0.0

    def initialize(self, simulation):
        """
        Initializes the model for simulation.
        """
        self.simulation = simulation
        mesh = self.simulation.cardiac_tissue.mesh
        self.u = self.init_u * np.ones(mesh.shape, dtype=simulation.npfloat)
        self.rhs = np.zeros_like(self.u)
        self.init_state_vars()

    def init_state_vars(self):
        for var in self.state_vars:
            val = getattr(self, "init_" + var)
            setattr(self, var, val * np.ones_like(self.u))

    def run(self):
        """
        Executes the ionic kernel for the Aliev-Panfilov model.
        """
        ionic_kernel(self.u, self.rhs,
                     self.simulation.cardiac_tissue.myo_indexes,
                     self.simulation.dt,
                     self.v, self.a, self.k, self.eap, self.mu_1, self.mu_2)


@njit(parallel=True)
def ionic_kernel(u, rhs, indexes, dt, v, a, k, eps, mu1, mu2, c_indexes=None):
    """
    Computes the ionic kernel for the Aliev-Panfilov 2D model.

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
        Recovery variable array.
    a : float
        Excitability threshold parameter.
    k : float
        Strength of the nonlinear source term (governs spike shape).
    eap : float
        Baseline recovery rate.
    mu_1 : float
        Recovery rate coefficient (scales v feedback).
    mu_2 : float
        Recovery rate offset (modulates u-dependence of recovery).
    c_indexes : np.ndarray
        Array of continuous indices where the kernel should be computed.
    """
    if c_indexes is None:
        c_indexes = indexes

    for i in prange(len(c_indexes)):
        m_i = indexes[i]
        c_i = c_indexes[i]
        v.flat[m_i] += dt * calc_dv(u.flat[c_i], v.flat[m_i], a, k, eps, mu1,
                                    mu2)
        rhs.flat[c_i] = dt * calc_rhs(u.flat[c_i], v.flat[m_i], a, k)


@njit
def calc_rhs(u, v, a, k) -> float:
    return - k * u * (u - a) * (u - 1.) - u * v


@njit
def calc_dv(u, v, a, k, eps, mu1, mu2) -> float:
    dv = (- (eps + (mu1 * v) / (mu2 + u)) * (v + k * u * (u - a - 1.)))
    return dv
