from ._cardiac_model import CardiacModel


class Courtemanche(CardiacModel):
    """
    This model describes the ionic currents and action potential dynamics of human atrial myocytes. 
    It includes detailed formulations for major ionic currents (fast sodium current, L-type calcium current, 
    inward rectifier potassium current, transient outward potassium current, rapid and slow delayed rectifier potassium currents, 
    and Na⁺/Ca²⁺ exchanger), as well as calcium handling mechanisms.

    The Courtemanche model is widely used as a reference atrial electrophysiology model. 
    It has served as the basis for many subsequent atrial modeling studies, including investigations of atrial fibrillation and drug effects.

    Model diffusion
    ---------------
    D_model : float
        Model-specific diffusion coefficient for simulating spatial propagation.

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
    model_name = "courtemanche"

    def __init__(self, memory_save=True):
        super().__init__()
        self.D_model = 0.154
