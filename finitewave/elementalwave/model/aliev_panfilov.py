import numpy as np
from numba import njit, prange

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.elementalwave.stencil.triangulated_isotropic_stencil import (
    TriangulatedIsotropicStencil
)


class AlievPanfilov(CardiacModel):
    def __init__(self):
        super().__init__()
        self.v = np.ndarray

        self.D_model = 1.

        self.state_vars = ["u", "v"]
        self.npfloat = 'float64'

        # model parameters
        self.a = 0.1
        self.k = 8.0
        self.eap = 0.01
        self.mu_1 = 0.2
        self.mu_2 = 0.3

        # initial conditions
        self.init_u = 0.0
        self.init_v = 0.0

    def initialize(self):
        super().initialize()
        self.u = self.init_u * np.ones_like(self.u, dtype=self.npfloat)
        self.v = self.init_v * np.ones_like(self.u, dtype=self.npfloat)
        self.rhs = np.zeros_like(self.u)

    def run_ionic_kernel(self):
        ionic_kernel(self.u_new,
                     self.u,
                     self.v,
                     self.cardiac_tissue.myo_indexes,
                     self.dt,
                     self.a,
                     self.k,
                     self.eap,
                     self.mu_1,
                     self.mu_2)

    def run_diffusion_kernel(self):
        self.diffusion_kernel()

    def select_stencil(self, cardiac_tissue):
        if cardiac_tissue.fibers is None:
            return TriangulatedIsotropicStencil

        return


@njit(parallel=True)
def ionic_kernel(u, v, rhs, indexes, dt, a, k, eap, mu1, mu2):
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
    for i in prange(indexes):
        v[i] += dt * calc_dv(v[i], u[i], a, k, eap, mu1, mu2)
        rhs[i] = calc_rhs(u[i], v[i], a, k)
    return v, rhs


def get_parameters() -> dict[str, float]:
    return {"a": 0.1, "k": 8.0, "eps": 0.01, "mu1": 0.2, "mu2": 0.3}


@njit
def calc_rhs(u, v, a, k):
    return - k * u * (u - a) * (u - 1.) - u * v


@njit
def calc_dv(v, u, a, k, eps, mu1, mu2):
    dv = (- (eps + (mu1 * v) / (mu2 + u)) * (v + k * u * (u - a - 1.)))
    return dv
