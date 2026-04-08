import math
import numpy as np

from ._cardiac_model import CardiacModel

from finitewave.cpuwave.model._registry import load_ops, wrap_calc


try:
    ops = load_ops("luo_rudy_91")
    jit_ops = wrap_calc(ops)
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

    Attributes
    ----------
    D_model : float
        Diffusion coefficient for the model, used in tissue simulations.
    npfloat : str
        String specifying the floating-point precision to use (e.g., 'float64').
    
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
        self.npfloat = "float64"
        self._initialize_variables_and_parameters(ops)
        self._initialize_model_func(jit_ops)

    def initialize(self, simulation):
        """
        Initializes the model for simulation.
        """
        super().initialize(simulation)

        self._initialize_ionic_kernel(ops.ionic_step, self.model_func)
        self.ionic_kernel_args = [getattr(self, name) for name in self.ionic_kernel_arg_names]

    def prepacing(self, stim_prepacing):
        self._initialize_prepacing_kernel(ops.ionic_step)
        return self._prepacing(stim_prepacing)

    def _initialize_model_func(self, jit_ops):
        self.model_func = {
            "calc_dm": jit_ops["calc_dm"],
            "calc_dh": jit_ops["calc_dh"],
            "calc_dj": jit_ops["calc_dj"],
            "calc_dd": jit_ops["calc_dd"],
            "calc_df": jit_ops["calc_df"],
            "calc_dx": jit_ops["calc_dx"],
            "calc_dcai": jit_ops["calc_dcai"],
            "calc_ina": jit_ops["calc_ina"],
            "calc_isk": jit_ops["calc_isk"],
            "calc_ik": jit_ops["calc_ik"],
            "calc_ik1": jit_ops["calc_ik1"],
            "calc_ikp": jit_ops["calc_ikp"],
            "calc_ib": jit_ops["calc_ib"],
            "calc_rhs": jit_ops["calc_rhs"],
        }
