import numpy as np
from numba import njit, prange

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import (
    AsymmetricStencil2D
)
from finitewave.cpuwave2D.stencil.isotropic_stencil_2d import (
    IsotropicStencil2D
)


class FentonKarma2D(CardiacModel):

    def __init__(self):
        """
        Initializes the Fenton-Karma instance with default parameters.

        Paper
        -----
        Fenton, F., & Karma, A. (1998).
        Vortex dynamics in three-dimensional continuous myocardium 
        with fiber rotation: Filament instability and fibrillationa
        Chaos, 8(1), 20-47.
        https://doi.org/10.1063/1.166311
            
        """
        super().__init__()
        self.v = np.ndarray
        self.w = np.ndarray

        self.D_model = 1.

        self.state_vars = ["u", "v", "w"]
        self.npfloat    = 'float64'

        # model parameters (MLR-I)
        
        self.tau_r   = 130
        self.tau_o   = 12.5
        self.tau_d   = 0.172
        self.tau_si  = 127
        self.tau_v_m = 18.2
        self.tau_v_p = 10
        self.tau_w_m = 80
        self.tau_w_p = 1020
        self.k       = 10
        self.u_c     = 0.13
        self.uc_si   = 0.85


    def initialize(self):
        """
        Initializes the model for simulation.
        """
        super().initialize()
        self.v = np.ones_like(self.u, dtype=self.npfloat)
        self.w = np.ones_like(self.u, dtype=self.npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel for the Fenton-Karma model.
        """
        ionic_kernel_2d(self.u_new, self.u, self.v, self.w, self.cardiac_tissue.myo_indexes, 
                        self.dt, self.tau_d, self.tau_o, self.tau_r, self.tau_si, 
                        self.tau_v_m, self.tau_v_p, self.tau_w_m, self.tau_w_p,
                        self.k, self.u_c, self.uc_si)

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
def calc_Jfi(u, v, u_c, tau_d):
    return (-v*(1 - u)*(u - u_c))/tau_d if (u - u_c) >= 0 else 0.0

@njit
def calc_Jso(u, u_c, tau_o, tau_r):
    jso_1 = u/tau_o if (u_c - u) >= 0 else 0.0 
    jso_2 = 1/tau_r if (u - u_c) >= 0 else 0.0
    return jso_1 + jso_2

@njit
def calc_Jsi(u, w, k, uc_si, tau_si):
    return -w*(1 + np.tanh(k*(u - uc_si)))/(2*tau_si)

@njit
def calc_v(v, u, dt, u_c, tau_v_m, tau_v_p):
    v_1 = (1 - v)/tau_v_m if (u_c - u) >= 0 else 0.0
    v_2 = v/tau_v_p if (u - u_c) >= 0 else 0.0
    v += dt*(v_1 - v_2)
    return v

@njit
def calc_w(w, u, dt, u_c, tau_w_m, tau_w_p):
    w_1 = (1 - w)/tau_w_m if (u_c - u) >= 0 else 0.0
    w_2 = w/tau_w_p if (u - u_c) >= 0 else 0.0
    w += dt*(w_1 - w_2)
    return w

@njit(parallel=True)
def ionic_kernel_2d(u_new, u, v, w, indexes, dt, 
                    tau_d, tau_o, tau_r, tau_si, 
                    tau_v_m, tau_v_p, tau_w_m, tau_w_p,
                    k, u_c, uc_si):
    """
    Computes the ionic kernel for the Fenton-Karma 2D model.

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

        v[i, j] = calc_v(v[i, j], u[i, j], dt, u_c, tau_v_m, tau_v_p)
        w[i, j] = calc_w(w[i, j], u[i, j], dt, u_c, tau_w_m, tau_w_p)

        J_fi = calc_Jfi(u[i, j], v[i, j], u_c, tau_d)
        J_so = calc_Jso(u[i, j], u_c, tau_o, tau_r)
        J_si = calc_Jsi(u[i, j], v[i, j], k, uc_si, tau_si)

        u_new[i, j] += dt * (-J_fi - J_so - J_si)

