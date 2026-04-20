from finitewave.mlxwave.model._cardiac_model import CardiacModel


class Barkley(CardiacModel):
    """
    Implementation of the Barkley model for excitable media.

    The Barkley model is a simplified two-variable reaction–diffusion system
    originally developed to study wave propagation in excitable media. While it is 
    not biophysically detailed, it captures essential qualitative features of 
    cardiac-like excitation dynamics such as spiral waves, wave break, and reentry.

    This implementation is included for benchmarking, educational purposes, 
    and comparison against more detailed cardiac models.

    Attributes
    ----------
    D_model : float
        Diffusion coefficient for excitation variable.
    npfloat : str
        Floating-point precision (default: 'float64').

    Model Variables
    ---------------
    u : np.ndarray
        Excitation variable (analogous to membrane potential).
    v : np.ndarray
        Recovery variable controlling excitability.
    
    Model Parameters
    ----------------
    a : float
        Threshold-like parameter controlling excitability.
    b : float
        Recovery time scale.
    eap : float
        Controls sharpness of the activation term (nonlinear gain).

    Paper
    -----
    Barkley, D. (1991).
    A model for fast computer simulation of waves in excitable media.
    Physica D: Nonlinear Phenomena, 61-70.
    https://doi.org/10.1016/0167-2789(86)90198-1.

    """
    model_name = "barkley"

    def __init__(self, memory_save=True):
        """Initializes the Barkley model instance with default parameters.

        Parameters
        ----------
        memory_save : bool
            Only for consistency with cpuwave version, mlx version always uses
            memory saving mode.
        """
        super().__init__()
        self.D_model = 1.0
