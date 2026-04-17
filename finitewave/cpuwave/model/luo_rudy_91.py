
from ._cardiac_model import CardiacModel

from .kernel._load_ops import load_ops


try:
    ops = load_ops("luo_rudy_91")
except KeyError as e:
    raise ImportError(
        "Luo–Rudy 1991 model ops not found."
        # "Install model package: pip install finitewave-model-luo-rudy91"
    ) from e


class LuoRudy91(CardiacModel):
    """
    Implements the Luo-Rudy 1991 ventricular action potential model.

    This biophysically detailed model simulates the ionic currents and membrane potential 
    of a ventricular cardiac cell based on Hodgkin-Huxley-type formalism. It was one of 
    the first to incorporate realistic ionic channel kinetics, calcium dynamics, and 
    multiple potassium currents to reproduce key phases of the action potential.

    The model includes:
    - Fast Na⁺ current (I_Na)
    - Slow inward Ca²⁺ current (I_Si)
    - Time-dependent K⁺ current (I_K)
    - Time-independent K⁺ current (I_K1)
    - Plateau K⁺ current (I_Kp)
    - Background/leak current (I_b)

    Model Diffusion
    ---------------
    D_model : float
        Model-specific diffusion coefficient for simulating spatial propagation.
    
    Model Variables
    ---------------
    u : np.ndarray
        Transmembrane potential in mV.
    m, h, j : np.ndarray
        Gating variables for the fast Na⁺ current.
    d, f : np.ndarray
        Gating variables for the slow inward Ca²⁺ current.
    x : np.ndarray
        Gating variable for the time-dependent K⁺ current.
    cai : np.ndarray
        Intracellular calcium concentration in mM.
        
    Model Parameters
    ----------------
    gna, gsi, gk, gk1, gkp, gb : float
        Maximum conductances for Na⁺, Ca²⁺, K⁺, and background channels [mS/μF].
    ko, ki, nao, nai, cao : float
        Ion concentrations in mM (extracellular and intracellular for Na⁺, K⁺, Ca²⁺).
    R, T, F : float
        Physical constants: gas constant, temperature in Kelvin, and Faraday constant.
    PR_NaK : float
        Sodium/potassium permeability ratio (used in reversal potential calculation for I_K).

    Paper
    -----
    Luo CH, Rudy Y. 
    A model of the ventricular cardiac action potential. Depolarization, repolarization, and their interaction. 
    Circ Res. 1991 Jun;68(6):1501-26. 
    doi: 10.1161/01.res.68.6.1501. 
    PMID: 1709839.

    """
    def __init__(self, memory_save=False):
        super().__init__(memory_save)

        self.D_model = 0.1
        self._initialize_variables_and_parameters(ops)
        self.ops = ops