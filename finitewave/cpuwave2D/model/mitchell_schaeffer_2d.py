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
        Implements the 2D Mitchell-Schaeffer model of cardiac excitation.

        This is a phenomenological two-variable model capturing the essence of cardiac 
        action potential dynamics using a simplified formulation. It separates inward and 
        outward currents and uses a single gating variable to regulate excitability.

        It reproduces key features like:
        - Excitability and recovery
        - Action potential duration (APD)
        - Restitution and wave propagation

        Attributes
        ----------
        h : np.ndarray
            Gating variable controlling the availability of inward current.
        D_model : float
            Diffusion coefficient for spatial propagation.
        state_vars : list
            Names of the dynamic variables for saving/restoring state.
        npfloat : str
            Floating-point type used (default: float64).

        Paper
        -----
        Mitchell, C. C., & Schaeffer, D. G. (2003).
        A two-current model for the dynamics of cardiac membrane
        potential. Bulletin of Mathematical Biology, 65, 767–793.
        https://doi.org/10.1016/S0092-8240(03)00041-7
            
        """
        super().__init__()
        self.h = np.ndarray

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
        self.h = np.ones_like(self.u, dtype=self.npfloat)

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
    Updates the gating variable h for the inward current.

    The gating variable h plays the role of a generic recovery mechanism.
    - It increases toward 1 with time constant tau_open when the membrane is at rest.
    - It decreases toward 0 with time constant tau_close when the membrane is excited.

    This mimics Na⁺ channel inactivation in a simplified way.

    Parameters
    ----------
    h : float
        Current value of the gating variable.
    u : float
        Membrane potential (dimensionless, in [0,1]).
    dt : float
        Time step [ms].
    tau_close : float
        Inactivation time constant (closing).
    tau_open : float
        Recovery time constant (opening).
    u_gate : float
        Threshold potential for switching gate dynamics.

    Returns
    -------
    float
        Updated value of h.
    """
    h += dt * (1.0 - h) / tau_open if u < u_gate else dt * (-h) / tau_close
    return h

@njit
def calc_J_in(h, u, tau_in):
    """
    Computes the inward current responsible for depolarization.

    This is a regenerative current:
    J_in = h * u² * (1 - u) / tau_in

    It activates when h is high (available) and u is sufficiently depolarized.
    The form ensures that the current grows with u but shuts off when u ~ 1.

    Parameters
    ----------
    h : float
        Gating variable controlling channel availability.
    u : float
        Membrane potential (dimensionless).
    tau_in : float
        Time constant for inward flow.

    Returns
    -------
    float
        Value of the inward current.
    """
    C = (u**2)*(1-u)
    return h*C/tau_in

@njit
def calc_J_out(u, tau_out):
    """
    Computes the outward current responsible for repolarization.

    This linear term simulates the slow repolarizing current that restores 
    the membrane potential back to rest.

    J_out = -u / tau_out

    Parameters
    ----------
    u : float
        Membrane potential.
    tau_out : float
        Time constant for outward current (repolarization).

    Returns
    -------
    float
        Value of the outward current.
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

