import numpy as np
from numba import njit, prange

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import (
    AsymmetricStencil2D
)
from finitewave.cpuwave2D.stencil.isotropic_stencil_2d import (
    IsotropicStencil2D
)


class MitchellSchaeffer2D(CardiacModel):

    def __init__(self):
        """
        Initializes the Mitchell-Schaeffer instance with default parameters.

        Paper
        -----
        Mitchell, C. C., & Schaeffer, D. G. (2003).
        A two-current model for the dynamics of cardiac membrane
        potential. Bulletin of Mathematical Biology, 65, 767–793.
        https://doi.org/10.1016/S0092-8240(03)00041-7
            
        """
        super().__init__()
        self.h = np.ndarray
        self.w = np.ndarray

        self.D_model = 1.

        self.state_vars = ["u", "h"]
        self.npfloat    = 'float64'

        # model parameters
        self.tau_close = 150
        self.tau_open  = 120
        self.tau_out = 6
        self.tau_in = 0.3
        self.u_gate = 0.13

    def initialize(self):
        """
        Initializes the model for simulation.
        """
        super().initialize()
        self.h = np.zeros_like(self.u, dtype=self.npfloat) + 1

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel for the Mitchell-Schaeffer model.
        """
        ionic_kernel_2d(self.u_new, self.u, self.h, self.cardiac_tissue.myo_indexes, self.dt, 
                        self.tau_close, self.tau_open, self.tau_in, self.tau_out, self.u_gate)

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
def calc_h(h, u, dt, tau_close, tau_open, u_gate):
    """
    Calculates the gating variable h for the Mitchell-Schaeffer model.

    Parameters
    ----------
    h : ndarray
        The h variable to be updated.
    u : ndarray
        The u variable representing the membrane potential.
    dt : float
        The time step for the simulation.
    tau_close : float
        The time constant for the closing of the h gate.
    tau_open : float
        The time constant for the opening of the h gate.
    u_gate : float
        The threshold value for the gating variable.
    """
    h += dt * (1.0 - h) / tau_open if u < u_gate else dt * (-h) / tau_close
    return h

@njit
def calc_J_in(h, u, tau_in):
    """
    Calculates the inward current for the Mitchell-Schaeffer model.

    Parameters
    ----------
    h : ndarray
        The h variable representing the gating variable.
    u : ndarray
        The u variable representing the membrane potential.
    tau_in : float
        The time constant for the inward current.
    """
    C = (u**2)*(1-u)
    return h*C/tau_in

@njit
def calc_J_out(u, tau_out):
    """
    Calculates the outward current for the Mitchell-Schaeffer model.

    Parameters
    ----------
    u : ndarray
        The u variable representing the membrane potential.
    tau_out : float
        The time constant for the outward current.
    """
    return -u/tau_out

@njit(parallel=True)
def ionic_kernel_2d(u_new, u, h, indexes, dt, tau_close, tau_open, tau_in, tau_out, u_gate):
    """
    Ionic kernel for the Mitchell-Schaeffer model in 2D.

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

    for ind in prange(len(indexes)):
        ii = indexes[ind]
        i = int(ii / n_j)
        j = ii % n_j

        h[i, j] = calc_h(h[i, j], u[i, j], dt, tau_close, tau_open, u_gate)

        J_in = calc_J_in(h[i, j], u[i, j], tau_in)
        J_out = calc_J_out(u[i, j], tau_out)
        u_new[i, j] += dt * (J_in + J_out)

