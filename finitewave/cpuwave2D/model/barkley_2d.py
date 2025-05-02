import numpy as np
from numba import njit, prange

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import (
    AsymmetricStencil2D
)
from finitewave.cpuwave2D.stencil.isotropic_stencil_2d import (
    IsotropicStencil2D
)


class Barkley2D(CardiacModel):
    """
    Implementation of the Barkley 2D cardiac model.

    Paper
    -----
    Barkley, D. (1991).
    A model for fast computer simulation of waves in excitable media.
    Physica D: Nonlinear Phenomena, 61-70.
    https://doi.org/10.1016/0167-2789(86)90198-1.

    """

    def __init__(self):
        """
        Initializes the Barkley2D instance with default parameters.
        """
        super().__init__()
        self.v = np.ndarray

        self.D_model = 1.
    
        self.state_vars = ["u", "v"]
        self.npfloat    = 'float64'

        # model parameters
        self.a    = 0.75
        self.b    = 0.02
        self.eap  = 0.02

    def initialize(self):
        """
        Initializes the model for simulation.
        """
        super().initialize()
        self.v = np.zeros_like(self.u, dtype=self.npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel for the Barkley model.
        """
        ionic_kernel_2d(self.u_new, self.u, self.v, self.cardiac_tissue.myo_indexes, self.dt, 
                        self.a, self.b, self.eap)

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
def calc_v(v, u, dt):
    """
    Calculates the recovery variable for the Barkley 2D model.

    Parameters
    ----------
    v : np.ndarray
        Recovery variable array.
    u : np.ndarray
        Action potential array.
    dt : float
        Time step for the simulation.

    Returns
    -------
    np.ndarray
        Updated recovery variable array.
    """

    v += dt*(u-v)
    return v


@njit(parallel=True)
def ionic_kernel_2d(u_new, u, v, indexes, dt, a, b, eap):
    """
    Computes the ionic kernel for the Barkley 2D model.

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

        v[i, j] = calc_v(v[i, j], u[i, j], dt)

        u_new[i, j] += dt * (u[i, j]*(1 - u[i, j])*(u[i, j] - (v[i, j] + b)/a))/eap
