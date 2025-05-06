from numba import njit, prange

from finitewave.cpuwave2D.model.barkley_2d import Barkley2D, calc_v
from finitewave.cpuwave3D.stencil.isotropic_stencil_3d import (
    IsotropicStencil3D
)
from finitewave.cpuwave3D.stencil.asymmetric_stencil_3d import (
    AsymmetricStencil3D
)


class Barkley3D(Barkley2D):
    """
    Implementation of the Barkley 3D cardiac model.

    See Barkley2D for the 2D model description.
    """

    def __init__(self):
        super().__init__()

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel for the Barkley model.
        """
        ionic_kernel_3d(self.u_new, self.u, self.v,
                        self.cardiac_tissue.myo_indexes, self.dt,
                        self.a, self.b, self.eap)

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
def ionic_kernel_3d(u_new, u, v, indexes, dt, a, b, eap):
    """
    Computes the ionic kernel for the Barkley 3D model.

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

    n_j = u.shape[1]
    n_k = u.shape[2]

    for ni in prange(len(indexes)):
        ii = indexes[ni]
        i = ii//(n_j*n_k)
        j = (ii % (n_j*n_k))//n_k
        k = (ii % (n_j*n_k)) % n_k

        v[i, j, k] = calc_v(v[i, j, k], u[i, j, k], dt)

        u_new[i, j, k] += dt * (u[i, j, k]*(1 - u[i, j, k])*(u[i, j, k] - (v[i, j, k] + b)/a))/eap

