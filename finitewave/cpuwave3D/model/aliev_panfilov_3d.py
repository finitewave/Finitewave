from numba import njit, prange

from finitewave.cpuwave2D.model.aliev_panfilov_2d import AlievPanfilov2D
from finitewave.cpuwave3D.stencil.isotropic_stencil_3d import (
    IsotropicStencil3D
)
from finitewave.cpuwave3D.stencil.asymmetric_stencil_3d import (
    AsymmetricStencil3D
)


class AlievPanfilov3D(AlievPanfilov2D):
    """
    Implementation of the Aliev-Panfilov 3D cardiac model.
    """

    def __init__(self):
        super().__init__()

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel for the Aliev-Panfilov model.
        """
        ionic_kernel_3d(self.u_new, self.u, self.v,
                        self.cardiac_tissue.myo_indexes, self.dt)

    def select_stencil(self, cardiac_tissue):
        """
        Selects the appropriate stencil for diffusion based on the tissue
        properties. If the tissue has fiber directions, an asymmetric stencil
        is used; otherwise, an isotropic stencil is used.

        Parameters
        ----------
        cardiac_tissue : CardiacTissue3D
            A 3D cardiac tissue object.

        Returns
        -------
        Stencil
            The stencil object to be used for diffusion computations.
        """

        if cardiac_tissue.fibers is None:
            return IsotropicStencil3D()

        return AsymmetricStencil3D()


@njit(parallel=True)
def ionic_kernel_3d(u_new, u, v, indexes, dt):
    """
    Computes the ionic kernel for the Aliev-Panfilov 3D model.

    Parameters
    ----------
    u_new : np.ndarray
        Array to store the updated action potential values.
    u : np.ndarray
        Current action potential array.
    v : np.ndarray
        Recovery variable array.
    dt : float
        Time step for the simulation.
    indexes : np.ndarray
        Array of indices where the kernel should be computed (``mesh == 1``).
    """
    # constants
    a = 0.1
    k_ = 8.
    eap = 0.01
    mu_1 = 0.2
    mu_2 = 0.3

    n_j = u.shape[1]
    n_k = u.shape[2]

    for ni in prange(len(indexes)):
        ii = indexes[ni]
        i = ii//(n_j*n_k)
        j = (ii % (n_j*n_k))//n_k
        k = (ii % (n_j*n_k)) % n_k

        u_new[i, j, k] += dt * (- k_ * u[i, j, k] * (u[i, j, k] - a) *
                                (u[i, j, k] - 1.) - u[i, j, k] * v[i, j, k])

        v[i, j, k] += (- dt * (eap + (mu_1 * v[i, j, k]) / (mu_2 + u[i, j, k]))
                       * (v[i, j, k] + k_ * u[i, j, k] * (u[i, j, k] - a - 1.))
                       )
