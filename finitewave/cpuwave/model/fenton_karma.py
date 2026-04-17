from ._cardiac_model import CardiacModel

from .kernel._load_ops import load_ops


try:
    ops = load_ops("fenton_karma")
except KeyError as e:
    raise ImportError(
        "Fenton-Karma model ops not found. "
        # "Install model package: pip install finitewave-model-fenton-karma"
    ) from e


class FentonKarma(CardiacModel):
    """
    Implementation of the Fenton-Karma model of cardiac electrophysiology.

    The Fenton-Karma model is a minimal three-variable model designed to reproduce
    essential features of human ventricular action potentials, including restitution, 
    conduction velocity dynamics, and spiral wave behavior. It captures the interaction 
    between fast depolarization, slow repolarization, and calcium-mediated effects 
    through simplified phenomenological equations.

    This implementation corresponds to the MLR-I parameter set described in the original paper
    and supports isotropic and anisotropic tissue simulations with diffusion.

    Attributes
    ----------
    D_model : float
        Baseline diffusion coefficient used in the diffusion stencil.
    npfloat : str
        Floating point precision (default is 'float64').

    Model Variables
    ---------------
    u : np.ndarray
        Transmembrane potential (normalized, dimensionless).
    v : np.ndarray
        Fast recovery variable, representing sodium channel inactivation.
    w : np.ndarray
        Slow recovery variable, representing calcium channel dynamics.
    
    Model Parameters
    ----------------
    tau_r : float
        Time constant for repolarization (outward current).
    tau_o : float
        Time constant for the open-state decay of fast sodium channels.
    tau_d : float
        Time constant for depolarization (fast inward current).
    tau_si : float
        Time constant for the slow inward (calcium-like) current.
    tau_v_m : float
        Time constant for inactivation gate v (membrane below threshold).
    tau_v_p : float
        Time constant for recovery gate v (above threshold).
    tau_w_m : float
        Time constant for recovery gate w (below threshold).
    tau_w_p : float
        Time constant for decay of w (above threshold).
    k : float
        Steepness parameter for the slow inward current.
    u_c : float
        Activation threshold for recovery dynamics.
    uc_si : float
        Activation threshold for the slow inward current.
    
    Paper
    -----
    Fenton, F., & Karma, A. (1998).
    Vortex dynamics in three-dimensional continuous myocardium 
    with fiber rotation: Filament instability and fibrillation.
    Chaos, 8(1), 20-47.
    https://doi.org/10.1063/1.166311
            
    """

    def __init__(self, memory_save=False):
        """
        Initializes the Fenton-Karma instance with default parameters.
        """
        super().__init__(memory_save)
        self.D_model = 0.1
        self._initialize_variables_and_parameters(ops)
        self.ops = ops
