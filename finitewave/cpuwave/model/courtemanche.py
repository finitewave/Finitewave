import math
import numpy as np

from ._cardiac_model import CardiacModel

from finitewave.cpuwave.model._registry import load_ops, wrap_calc
from finitewave.cpuwave.model._kernel_builder import build_kernel


try:
    ops = load_ops("courtemanche")
    jit_ops = wrap_calc(ops)
except KeyError as e:
    raise ImportError(
        "Courtemanche model ops not found. "
        "Install model package: pip install finitewave-model-courtemanche"
    ) from e


class Courtemanche(CardiacModel):
    """
    This model describes the ionic currents and action potential dynamics of human atrial myocytes. 
    It includes detailed formulations for major ionic currents (fast sodium current, L-type calcium current, 
    inward rectifier potassium current, transient outward potassium current, rapid and slow delayed rectifier potassium currents, 
    and Na⁺/Ca²⁺ exchanger), as well as calcium handling mechanisms.

    The Courtemanche model is widely used as a reference atrial electrophysiology model. 
    It has served as the basis for many subsequent atrial modeling studies, including investigations of atrial fibrillation and drug effects.

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
    nai : np.ndarray
        Intracellular sodium concentration in mM.
    ki : np.ndarray
        Intracellular potassium concentration in mM.
    cai : np.ndarray
        Intracellular calcium concentration in mM.
    m, h, j : np.ndarray
        Gating variables for the fast sodium current.
    oa, oi : np.ndarray
        Gating variables for the transient outward potassium current (I_to).
    ua, ui : np.ndarray
        Gating variables for the ultrarapid delayed rectifier potassium current (I_kur).
    xr : np.ndarray
        Gating variable for the rapid delayed rectifier potassium current (I_kr).
    xs : np.ndarray
        Gating variable for the slow delayed rectifier potassium current (I_ks).
    d, f : np.ndarray
        Gating variables for the L-type calcium current (I_cal).
    fca : np.ndarray
        Calcium-dependent inactivation variable for I_cal.
    urel, vrel, wrel : np.ndarray
        Gating variables for sarcoplasmic reticulum (SR) calcium release.
    ire : np.ndarray
        SR calcium release current.
    caup : np.ndarray
        SR calcium uptake variable.
    carel : np.ndarray
        SR calcium release variable.

    Model Parameters
    ----------------
    nao : float
        Extracellular sodium concentration in mM.
    ko : float
        Extracellular potassium concentration in mM.
    cao : float
        Extracellular calcium concentration in mM.
    R : float
        Universal gas constant in J/(mol*K).
    T : float
        Absolute temperature in K.
    F : float
        Faraday's constant in C/mol.
    Cm : float
        Membrane capacitance in μF/cm².
    gna : float
        Maximum conductance for the fast sodium current in mS/μF.
    gk1 : float
        Maximum conductance for the inward rectifier potassium current in mS/μF.
    gto : float
        Maximum conductance for the transient outward potassium current in mS/μF.
    gcal : float
        Maximum conductance for the L-type calcium current in mS/μF.
    gnab : float
        Maximum conductance for the background sodium current in mS/μF.
    gcab : float
        Maximum conductance for the background calcium current in mS/μF.
    gkr : float
        Maximum conductance for the rapid delayed rectifier potassium current in mS/μF.
    gks : float
        Maximum conductance for the slow delayed rectifier potassium current in mS/μF.
    inakmax : float
        Maximum current for the Na⁺/K⁺ pump in μA/μF.
    kmnai : float
        Half-saturation concentration for intracellular sodium in mM (Na⁺/K⁺ pump).
    kmko : float
        Half-saturation concentration for extracellular potassium in mM (Na⁺/K⁺ pump).
    inacamax : float
        Maximum current for the Na⁺/Ca²⁺ exchanger in μA/μF.
    kmnancx : float
        Half-saturation concentration for intracellular sodium in mM (Na⁺/Ca²⁺ exchanger).
    kmcancx : float
        Half-saturation concentration for intracellular calcium in mM (Na⁺/Ca²⁺ exchanger).
    ksatncx : float
        Saturation factor for the Na⁺/Ca²⁺ exchanger.
    ipcamax : float
        Maximum current for the plasma membrane Ca²⁺ ATPase in μA/μF.
    iupmax : float
        Maximum uptake rate for the SR calcium pump in mM/ms.
    kup : float
        Half-saturation concentration for calcium uptake in mM.
    caupmax : float
        Maximum SR calcium uptake variable in mM.
    krel : float
        Rate constant for SR calcium release in ms⁻¹.
    Vrel : float
        Volume ratio for SR release.
    Vup : float
        Volume ratio for SR uptake.
    Vj : float
        Volume ratio for junctional space.
    kq10 : float
        Temperature coefficient for gating kinetics.
    ibk : float
        Background potassium current in μA/μF.
    trpnmax : float
        Maximum concentration of troponin C in mM.
    kmtrpn : float
        Half-saturation concentration for troponin C in mM.
    cmdnmax : float
        Maximum concentration of calmodulin in mM.
    kmcmdn : float
        Half-saturation concentration for calmodulin in mM.
    csqnmax : float
        Maximum concentration of calsequestrin in mM.
    kmcsqn : float
        Half-saturation concentration for calsequestrin in mM.
    

    Paper
    -----
    Courtemanche M, Ramirez RJ, Nattel S. 
    Ionic mechanisms underlying human atrial action potential properties: insights from a mathematical model. 
    Am J Physiol. 1998 Jul;275(1):H301-21.
    https://doi.org/10.1152/ajpheart.1998.275.1.H301
    """
    def __init__(self, memory_save=False):
        super().__init__(memory_save)
        self.D_model = 0.154
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
            "calc_gating_variable_rush_larsen": jit_ops["calc_gating_variable_rush_larsen"],
            "calc_ena": jit_ops["calc_ena"],
            "calc_ek": jit_ops["calc_ek"],
            "calc_eca": jit_ops["calc_eca"],
            "calc_am": jit_ops["calc_am"],
            "calc_bm": jit_ops["calc_bm"],
            "calc_tau": jit_ops["calc_tau"],
            "calc_inf": jit_ops["calc_inf"],
            "calc_ah": jit_ops["calc_ah"],
            "calc_bh": jit_ops["calc_bh"],
            "calc_aj": jit_ops["calc_aj"],
            "calc_bj": jit_ops["calc_bj"],
            "calc_tau_oa": jit_ops["calc_tau_oa"],
            "calc_oa_inf": jit_ops["calc_oa_inf"],
            "calc_tau_oi": jit_ops["calc_tau_oi"],
            "calc_oi_inf": jit_ops["calc_oi_inf"],
            "calc_tau_ua": jit_ops["calc_tau_ua"],
            "calc_ua_inf": jit_ops["calc_ua_inf"],
            "calc_tau_ui": jit_ops["calc_tau_ui"],
            "calc_ui_inf": jit_ops["calc_ui_inf"],
            "calc_tau_xr": jit_ops["calc_tau_xr"],
            "calc_xr_inf": jit_ops["calc_xr_inf"],
            "calc_tau_xs": jit_ops["calc_tau_xs"],
            "calc_xs_inf": jit_ops["calc_xs_inf"],
            "calc_tau_d": jit_ops["calc_tau_d"],
            "calc_d_inf": jit_ops["calc_d_inf"],
            "calc_tau_f": jit_ops["calc_tau_f"],
            "calc_f_inf": jit_ops["calc_f_inf"],
            "calc_tau_fca": jit_ops["calc_tau_fca"],
            "calc_fca_inf": jit_ops["calc_fca_inf"],
            "calc_ina": jit_ops["calc_ina"],
            "calc_ik1": jit_ops["calc_ik1"],
            "calc_ito": jit_ops["calc_ito"],
            "calc_ikur": jit_ops["calc_ikur"],
            "calc_ikr": jit_ops["calc_ikr"],
            "calc_iks": jit_ops["calc_iks"],
            "calc_ical": jit_ops["calc_ical"],
            "calc_inak": jit_ops["calc_inak"],
            "calc_inaca": jit_ops["calc_inaca"],
            "calc_ibca": jit_ops["calc_ibca"],
            "calc_ibna": jit_ops["calc_ibna"],
            "calc_ipca": jit_ops["calc_ipca"],
            "calc_Fn": jit_ops["calc_Fn"],
            "calc_tau_urel": jit_ops["calc_tau_urel"],
            "calc_urel_inf": jit_ops["calc_urel_inf"],
            "calc_tau_vrel": jit_ops["calc_tau_vrel"],
            "calc_vrel_inf": jit_ops["calc_vrel_inf"],
            "calc_tau_wrel": jit_ops["calc_tau_wrel"],
            "calc_wrel_inf": jit_ops["calc_wrel_inf"],
            "calc_irel": jit_ops["calc_irel"],
            "calc_itr": jit_ops["calc_itr"],
            "calc_iup": jit_ops["calc_iup"],
            "calc_iupleak": jit_ops["calc_iupleak"],
            "calc_dcaup": jit_ops["calc_dcaup"],
            "calc_dnai": jit_ops["calc_dnai"],
            "calc_dki": jit_ops["calc_dki"],
            "calc_dcai": jit_ops["calc_dcai"],
            "calc_dcarel": jit_ops["calc_dcarel"],           
            "calc_rhs": jit_ops["calc_rhs"],
        }
