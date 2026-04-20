from ._cardiac_model import CardiacModel


class MitchellSchaeffer(CardiacModel):
    """
    Implements the Mitchell-Schaeffer model of cardiac excitation.

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

    model_name = "mitchell_schaeffer"

    def __init__(self, memory_save=True):
        """
        Initializes the Mitchell-Schaeffer instance with default parameters.

        Parameters
        ----------
        memory_save : bool
            Only for consistency with cpuwave version, mlx version always uses
            memory saving mode.
        """
        super().__init__()
        self.D_model = 1.
