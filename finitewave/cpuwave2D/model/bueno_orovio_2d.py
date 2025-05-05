import numpy as np
from numba import njit, prange

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import (
    AsymmetricStencil2D
)
from finitewave.cpuwave2D.stencil.isotropic_stencil_2d import (
    IsotropicStencil2D
)


class BuenoOrovio2D(CardiacModel):

    def __init__(self):
        """
        Initializes the Bueno-Orovio instance with default parameters.

        Paper
        -----
        Bueno-Orovio, A., Cherry, E. M., & Fenton, F. H. (2008).
        Minimal model for human ventricular action potentials in tissue.
        J Theor Biol., 253(3), 544-60.
        https://doi.org/10.1016/j.jtbi.2008.03.029

            
        """
        super().__init__()
        self.v = np.ndarray
        self.w = np.ndarray
        self.s = np.ndarray

        self.D_model = 1.

        self.state_vars = ["u", "v", "w", "s"]
        self.npfloat    = 'float64'

        # model parameters (EPI)
        
        self.u_o = 0.0
        self.u_u = 1.55
        self.theta_v = 0.3
        self.theta_w = 0.13
        self.theta_v_m = 0.006
        self.theta_o = 0.006
        self.tau_v1_m = 60
        self.tau_v2_m = 1150
        self.tau_v_p = 1.4506
        self.tau_w1_m = 60
        self.tau_w2_m = 15
        self.k_w_m = 65
        self.u_w_m = 0.03
        self.tau_w_p = 200
        self.tau_fi = 0.11
        self.tau_o1 = 400
        self.tau_o2 = 6
        self.tau_so1 = 30.0181
        self.tau_so2 = 0.9957
        self.k_so = 2.0458
        self.u_so = 0.65
        self.tau_s1 = 2.7342
        self.tau_s2 = 16
        self.k_s = 2.0994
        self.u_s = 0.9087
        self.tau_si = 1.8875
        self.tau_w_inf = 0.07
        self.w_inf_ = 0.94

    def initialize(self):
        """
        Initializes the model for simulation.
        """
        super().initialize()
        self.v = np.ones_like(self.u, dtype=self.npfloat)
        self.w = np.ones_like(self.u, dtype=self.npfloat)
        self.s = np.zeros_like(self.u, dtype=self.npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel for the Bueno-Orovio model.
        """
        ionic_kernel_2d(self.u_new, self.u, self.v, self.w, self.s, self.cardiac_tissue.myo_indexes, 
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
def calc_v(v, u, dt, theta_v, v_inf, tau_v_m, tau_v_p):
    v_ = (v_inf - v)/tau_v_m if (u - theta_v) < 0 else -v/tau_v_p
    v += dt*v_
    return v

@njit
def calc_w(w, u, dt, theta_w, w_inf, tau_w_m, tau_w_p):
    w_ = (w_inf - w)/tau_w_m if (u - theta_w) < 0 else -w/tau_w_p
    w += dt*w_
    return w

@njit
def calc_s(s, u, dt, tau_s, k_s, u_s):
    s += dt*((1 + np.tanh(k_s*(u - u_s)))/2 - s)/tau_s
    return s

@njit
def calc_Jfi(u, v, theta_v, u_u, tau_fi):
    H = 1.0 if (u - theta_v) >= 0 else 0.0
    return -v*H*(u - theta_v)*(u_u - u)/tau_fi

@njit
def calc_Jso(u, u_o, theta_w, tau_o, tau_so):
    H = 1.0 if (u - theta_w) >= 0 else 0.0
    return (u - u_o)*(1 - H)/tau_o + H/tau_so

@njit
def calc_Jsi(u, w, s, theta_w, tau_si):
    H = 1.0 if (u - theta_w) >= 0 else 0.0
    return -H*w*s/tau_si

@njit
def calc_tau_v_m(u, theta_v_m, tau_v1_m, tau_v2_m):
    return tau_v1_m if (u - theta_v_m) < 0 else tau_v2_m

@njit    
def calc_tau_w_m(u, tau_w1_m, tau_w2_m, k_w_m, u_w_m):
    return tau_w1_m + (tau_w2_m - tau_w1_m)*(1 + np.tanh(k_w_m*(u - u_w_m)))/2

@njit
def calc_tau_so(u, tau_so1, tau_so2, k_so, u_so):
    return tau_so1 + (tau_so2 - tau_so1)*(1 + np.tanh(k_so*(u - u_so)))/2

@njit
def calc_tau_s(u, tau_s1, tau_s2, theta_w):
    return tau_s1 if (u - theta_w) < 0 else tau_s2

@njit
def calc_tau_o(u, tau_o1, tau_o2, theta_o):
    return tau_o1 if (u - theta_o) < 0 else tau_o2
    
@njit
def calc_v_inf(u, theta_v_m):
    return 1.0 if u < theta_v_m else 0.0

@njit
def calc_w_inf(u, theta_o, tau_w_inf, w_inf_):
    return 1 - u/tau_w_inf if (u - theta_o) < 0 else w_inf_


@njit(parallel=True)
def ionic_kernel_2d(u_new, u, v, w, s, indexes, dt, 
                    u_o, u_u, theta_v, theta_w, theta_v_m,
                    theta_o, tau_v1_m, tau_v2_m, tau_v_p,
                    tau_w1_m, tau_w2_m, k_w_m, u_w_m,
                    tau_w_p, tau_fi, tau_o1, tau_o2,
                    tau_so1, tau_so2, k_so, u_so,
                    tau_s1, tau_s2, k_s, u_s,
                    tau_si, tau_w_inf, w_inf_):
    """
    Computes the ionic kernel for the Bueno-Orovio 2D model.

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

        v[i, j] = calc_v(v[i, j], u[i, j], dt, theta_v, calc_v_inf(u[i, j], theta_v_m), 
                         calc_tau_v_m(u[i, j], theta_v_m, tau_v1_m, tau_v2_m), tau_v_p)
        
        w[i, j] = calc_w(w[i, j], u[i, j], dt, theta_w, calc_w_inf(u[i, j], theta_o, tau_w_inf, w_inf_), 
                         calc_tau_w_m(u[i, j], tau_w1_m, tau_w2_m, k_w_m, u_w_m), tau_w_p)
        
        s[i, j] = calc_s(s[i, j], u[i, j], dt,
                        calc_tau_s(u[i, j], tau_s1, tau_s2, theta_w), k_s, u_s)
        
        J_fi = calc_Jfi(u[i, j], v[i, j], theta_v, u_u, tau_fi)
        J_so = calc_Jso(u[i, j], u_o, theta_w,
                        calc_tau_o(u[i, j], tau_o1, tau_o2, theta_o), 
                        calc_tau_so(u[i, j], tau_so1, tau_so2, k_so, u_so))
        J_si = calc_Jsi(u[i, j], w[i, j], s[i, j], theta_w, tau_si)

        u_new[i, j] += dt * (-J_fi - J_so - J_si)