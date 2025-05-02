from numba import njit, prange

from finitewave.cpuwave2D.model.fenton_karma_2d import (
    FentonKarma2D,
    calc_Jfi,
    calc_Jsi,
    calc_Jso,
    calc_v,
    calc_w,
)
from finitewave.cpuwave3D.stencil.isotropic_stencil_3d import (
    IsotropicStencil3D
)
from finitewave.cpuwave3D.stencil.asymmetric_stencil_3d import (
    AsymmetricStencil3D
)


class FentonKarma3D(FentonKarma2D):
    """
    Implementation of the Fenton-Karma 3D cardiac model.
    """

    def __init__(self):
        super().__init__()

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel for the Fenton-Karma model.
        """
        ionic_kernel_3d(self.u_new, self.u, self.v, self.w, self.cardiac_tissue.myo_indexes, 
                        self.dt, self.u_c, self.tau_d, self.tau_o, self.tau_r, self.tau_si, 
                        self.tau_v_m, self.tau_v_p, self.tau_w_m, self.tau_w_p)

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
def ionic_kernel_3d(u_new, u, v, w, indexes, dt, 
                    tau_d, tau_o, tau_r, tau_si, 
                    tau_v_m, tau_v_p, tau_w_m, tau_w_p,
                    k, u_c, uc_si):
    """
    Computes the ionic kernel for the Fenton-Karma 3D model.

    Parameters
    ----------
    u_new : ndarray
        The new state of the u variable.
    u : ndarray
        The current state of the u variable.
    myo_indexes : list
        List of indexes representing myocardial cells.
    dt : float
        The time step for the simulation.
    """

    n_j = u.shape[1]
    n_k = u.shape[2]

    for ni in prange(len(indexes)):
        ii = indexes[ni]
        i = ii//(n_j*n_k)
        j = (ii % (n_j*n_k))//n_k
        k_ = (ii % (n_j*n_k)) % n_k
        
        J_fi = calc_Jfi(u[i, j, k_], v[i, j, k_], u_c, tau_d)
        J_so = calc_Jso(u[i, j, k_], u_c, tau_o, tau_r)
        J_si = calc_Jsi(u[i, j, k_], v[i, j, k_], k, uc_si, tau_si)

        v[i, j, k_] = calc_v(v[i, j, k_], u[i, j, k_], u_c, tau_v_m, tau_v_p)
        w[i, j, k_] = calc_w(w[i, j, k_], u[i, j, k_], u_c, tau_w_m, tau_w_p)

        u_new[i, j, k_] += dt * (-J_fi - J_so - J_si)

