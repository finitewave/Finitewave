import numpy as np
from numba import njit, prange

from .cardiac_model_elems import CardiacModelElems


class AlievPanfilovElems(CardiacModelElems):
    def __init__(self):
        super().__init__()

        self.state_vars = ["u", "v"]

        self.a = 0.1
        self.k = 8.0
        self.eps = 0.01
        self.mu1 = 0.2
        self.mu2 = 0.3
        self.D_model = 1.

    def initialize(self):
        super().initialize()
        self.v = np.zeros_like(self.u)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel for the Aliev-Panfilov model.
        """
        ionic_kernel_elems(self._u, self.v, self._rhs,
                           self.cardiac_tissue.myo_indexes, self.dt,
                           self.a, self.k, self.eps, self.mu1, self.mu2)


@njit(parallel=True)
def ionic_kernel_elems(u, v, rhs, indexes, dt, a, k, eps, mu1, mu2):
    """
    Computes the ionic kernel for the Aliev-Panfilov 2D model.

    Parameters
    ----------
    u_new : np.ndarray
        Array to store the updated action potential values.
    u : np.ndarray
        Current action potential array.
    v : np.ndarray
        Recovery variable array.
    indexes : np.ndarray
        Array of indices where the kernel should be computed (``mesh == 1``).
    dt : float
        Time step for the simulation.
    """
    for i in prange(len(indexes)):
        ind = indexes[i]
        v[ind] += dt * calc_dv(v[ind], u[i], a, k, eps, mu1, mu2)
        rhs[i] = dt * calc_rhs(u[i], v[ind], a, k)

    # v[indexes] += dt * calc_dv(v[indexes], u[indexes], a, k, eps, mu1, mu2)
    # rhs[indexes] = dt * calc_rhs(u[indexes], v[indexes], a, k)


@njit
def calc_rhs(u, v, a, k) -> float:
    return - k * u * (u - a) * (u - 1.) - u * v


@njit
def calc_dv(v, u, a, k, eps, mu1, mu2) -> float:
    dv = (- (eps + (mu1 * v) / (mu2 + u)) * (v + k * u * (u - a - 1.)))
    return dv
