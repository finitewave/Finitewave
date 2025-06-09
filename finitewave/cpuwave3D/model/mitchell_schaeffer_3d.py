from numba import njit, prange

from finitewave.cpuwave2D.model.mitchell_schaeffer_2d import (
    MitchellSchaeffer2D, 
    calc_h, 
    calc_J_in, 
    calc_J_out
)
from finitewave.cpuwave3D.stencil.isotropic_stencil_3d import (
    IsotropicStencil3D
)
from finitewave.cpuwave3D.stencil.asymmetric_stencil_3d import (
    AsymmetricStencil3D
)


class MitchellSchaeffer3D(MitchellSchaeffer2D):
    """
    Implementation of the Mitchell-Schaeffer 3D cardiac model.

    See MitchellSchaeffer2D for the 2D model description.
    """

    def __init__(self):
        super().__init__()

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel for the Mitchell-Schaeffer model.
        """
        ionic_kernel_3d(self.u_new, self.u, self.h, self.cardiac_tissue.myo_indexes, self.dt, 
                        self.tau_close, self.tau_open, self.tau_in, self.tau_out, self.u_gate)

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
def ionic_kernel_3d(u_new, u, h, indexes, dt, tau_close, tau_open, tau_in, tau_out, u_gate):
    """
    Computes the ionic kernel for the Mitchell-Schaeffer 3D model.

    Parameters
    ----------
    u_new : ndarray
        The new state of the u variable.
    u : ndarray
        The current state of the u variable.
    h : ndarray
        The gating variable h.
    myo_indexes : list
        List of indexes representing myocardial cells.
    dt : float
        The time step for the simulation.
    tau_close : float
        The time constant for the closing of the h gate.
    tau_open : float
        The time constant for the opening of the h gate.
    u_gate : float
        The threshold value for the gating variable.
    """

    n_j = u.shape[1]
    n_k = u.shape[2]

    for ni in prange(len(indexes)):
        ii = indexes[ni]
        i = ii//(n_j*n_k)
        j = (ii % (n_j*n_k))//n_k
        k = (ii % (n_j*n_k)) % n_k
        
        h[i, j, k] = calc_h(h[i, j, k], u[i, j, k], dt, tau_close, tau_open, u_gate)

        J_in = calc_J_in(h[i, j, k], u[i, j, k], tau_in)
        J_out = calc_J_out(u[i, j, k], tau_out)
        u_new[i, j, k] += dt * (J_in + J_out)

