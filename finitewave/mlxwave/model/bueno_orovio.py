
from ._cardiac_model import CardiacModel

from .kernel._load_ops import load_ops


try:
    ops = load_ops("bueno_orovio")
except KeyError as e:
    raise ImportError(
        "Bueno-Orovio model ops not found. "
        # "Install model package: pip install finitewave-model-bueno-orovio"
    ) from e


class BuenoOrovio(CardiacModel):
    """
    Implementation of the Bueno-Orovio–Cherry–Fenton (BOCF) model 
    for simulating human ventricular tissue electrophysiology.

    The BOCF model is a minimal phenomenological model developed to capture 
    key ionic mechanisms and reproduce realistic human ventricular action potential 
    dynamics, including restitution, conduction block, and spiral wave behavior. 
    It consists of four variables: transmembrane potential (u), two gating variables (v, w), 
    and one additional slow variable (s), representing calcium-related dynamics.

    This implementation corresponds to the EPI (epicardial) parameter set described in the paper.

    Attributes
    ----------
    D_model : float
        Diffusion coefficient for spatial propagation.
    npfloat : str
        Floating point precision (default: 'float64').

    Model Variables
    ---------------
    u : np.ndarray
        Transmembrane potential (dimensionless).
    v : np.ndarray
        Fast gating variable representing sodium channel inactivation.
    w : np.ndarray
        Slow recovery variable representing calcium and potassium gating.
    s : np.ndarray
        Slow variable related to calcium inactivation.
    
    Model Parameters (EPI set)
    --------------------------
    u_o : float
        Resting membrane potential.
    u_u : float
        Peak potential (upper bound).
    theta_v, theta_w : float
        Activation thresholds for v and w.
    theta_v_m, theta_o : float
        Thresholds for switching time constants.
    tau_v1_m, tau_v2_m : float
        Time constants for v below/above threshold.
    tau_v_p : float
        Decay constant for v.
    tau_w1_m, tau_w2_m : float
        Base and transition time constants for w.
    k_w_m, u_w_m : float
        Parameters controlling the shape of τw curve.
    tau_w_p : float
        Time constant for decay of w above threshold.
    tau_fi : float
        Time constant for fast inward current (J_fi).
    tau_o1, tau_o2 : float
        Time constants for outward current below/above threshold.
    tau_so1, tau_so2 : float
        Time constants for repolarizing tail current.
    k_so, u_so : float
        Parameters controlling nonlinearity in tau_so.
    tau_s1, tau_s2 : float
        Time constants for the s-gate below/above threshold.
    k_s, u_s : float
        Parameters for tanh activation of the s variable.
    tau_si : float
        Time constant for slow inward current (J_si).
    tau_w_inf : float
        Slope of w∞ below threshold.
    w_inf_ : float
        Asymptotic value of w∞ above threshold.

    Paper
    -----
    Bueno-Orovio, A., Cherry, E. M., & Fenton, F. H. (2008).
    Minimal model for human ventricular action potentials in tissue.
    J Theor Biol., 253(3), 544-60.
    https://doi.org/10.1016/j.jtbi.2008.03.029

    """

    def __init__(self):
        """
        Initializes the Bueno-Orovio instance with default parameters.
        """
        super().__init__()
        self.D_model = 0.1171
        self.ops = ops
        self._initialize_variables_and_parameters(ops)