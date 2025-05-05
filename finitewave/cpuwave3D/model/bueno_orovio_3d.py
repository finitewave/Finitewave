from numba import njit, prange

from finitewave.cpuwave2D.model.bueno_orovio_2d import (
    BuenoOrovio2D,
    calc_Jfi,
    calc_Jsi,
    calc_Jso,
    calc_tau_v_m,
    calc_tau_w_m,
    calc_tau_so,
    calc_tau_s,
    calc_tau_o,
    calc_v_inf,
    calc_w_inf,
    calc_v,
    calc_w,
    calc_s
)
from finitewave.cpuwave3D.stencil.isotropic_stencil_3d import (
    IsotropicStencil3D
)
from finitewave.cpuwave3D.stencil.asymmetric_stencil_3d import (
    AsymmetricStencil3D
)


class BuenoOrovio3D(BuenoOrovio2D):
    """
    Implementation of the Bueno-Orovio 3D cardiac model.
    """

    def __init__(self):
        super().__init__()

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel for the Bueno-Orovio model.
        """
        ionic_kernel_3d(self.u_new, self.u, self.v, self.w, self.s, self.cardiac_tissue.myo_indexes, 
                        self.dt, self.u_o, self.u_u, self.theta_v, self.theta_w, self.theta_v_m,
                        self.theta_o, self.tau_v1_m, self.tau_v2_m, self.tau_v_p,
                        self.tau_w1_m, self.tau_w2_m, self.k_w_m, self.u_w_m,
                        self.tau_w_p, self.tau_fi, self.tau_o1, self.tau_o2,
                        self.tau_so1, self.tau_so2, self.k_so, self.u_so,
                        self.tau_s1, self.tau_s2, self.k_s, self.u_s,
                        self.tau_si, self.tau_w_inf, self.w_inf_)

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
def ionic_kernel_3d(u_new, u, v, w, s, indexes, dt, 
                    u_o, u_u, theta_v, theta_w, theta_v_m,
                    theta_o, tau_v1_m, tau_v2_m, tau_v_p,
                    tau_w1_m, tau_w2_m, k_w_m, u_w_m,
                    tau_w_p, tau_fi, tau_o1, tau_o2,
                    tau_so1, tau_so2, k_so, u_so,
                    tau_s1, tau_s2, k_s, u_s,
                    tau_si, tau_w_inf, w_inf_):
    """
    Computes the ionic kernel for the Bueno-Orovio 3D model.

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
        

        v[i, j, k_] = calc_v(v[i, j, k_], u[i, j, k_], dt, theta_v, calc_v_inf(u[i, j, k_], theta_v_m), 
                         calc_tau_v_m(u[i, j, k_], theta_v_m, tau_v1_m, tau_v2_m), tau_v_p)
        
        w[i, j, k_] = calc_w(w[i, j, k_], u[i, j, k_], dt, theta_w, calc_w_inf(u[i, j, k_], theta_o, tau_w_inf, w_inf_), 
                         calc_tau_w_m(u[i, j, k_], tau_w1_m, tau_w2_m, k_w_m, u_w_m), tau_w_p)
        
        s[i, j, k_] = calc_s(s[i, j, k_], u[i, j, k_], dt,
                        calc_tau_s(u[i, j, k_], tau_s1, tau_s2, theta_w), k_s, u_s)
        
        J_fi = calc_Jfi(u[i, j, k_], v[i, j, k_], theta_v, u_u, tau_fi)
        J_so = calc_Jso(u[i, j, k_], u[i, j, k_], theta_w,
                        calc_tau_o(u[i, j, k_], tau_o1, tau_o2, theta_o), 
                        calc_tau_so(u[i, j, k_], tau_so1, tau_so2, k_so, u_so))
        J_si = calc_Jsi(u[i, j, k_], w[i, j, k_], s[i, j, k_], theta_w, tau_si)

        u_new[i, j, k_] += dt * (-J_fi - J_so - J_si)

