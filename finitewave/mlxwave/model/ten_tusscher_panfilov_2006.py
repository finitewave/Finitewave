from ._cardiac_model import CardiacModel


class TenTusscherPanfilov2006(CardiacModel):
    """
    Implements the ten Tusscher–Panfilov 2006 (TP06) human ventricular ionic model in 2D.

    The TP06 model is a detailed biophysical model of the human ventricular 
    action potential, designed to simulate realistic electrical behavior in 
    tissue including alternans, reentrant waves, and spiral wave breakup.

    This model includes:
    - Dynamic state variables (voltage, ion concentrations, channel gates, buffers)
    - Full calcium handling with subspace (cass) and sarcoplasmic reticulum (casr)
    - Sodium, potassium, and calcium currents including background, exchanger, and pumps
    - Buffering effects and intracellular transport

    Attributes
    ----------
    D_model : float
        Diffusion coefficient specific to this model (cm²/ms).
    npfloat : str
        String specifying the floating-point precision to use (e.g., 'float64').

    Model Variables
    ---------------
    u : np.ndarray
        Transmembrane potential (mV).
    cai : np.ndarray
        Intracellular calcium concentration (mM).
    casr : np.ndarray
        Calcium concentration in the sarcoplasmic reticulum (mM).
    cass : np.ndarray
        Calcium concentration in the subsarcolemmal space (mM).
    nai : np.ndarray
        Intracellular sodium concentration (mM).
    Ki : np.ndarray
        Intracellular potassium concentration (mM).
    m, h, j : np.ndarray
        Gating variables for the fast sodium current.
    xr1, xr2 : np.ndarray
        Gating variables for the rapid delayed rectifier K⁺ current.
    xs : np.ndarray
        Gating variable for the slow delayed rectifier K⁺ current.
    r, s : np.ndarray
        Gating variables for the transient outward K⁺ current.
    d, f, f2, fcass : np.ndarray
        Gating variables for the L-type calcium current.
    rr, oo : np.ndarray
        Gating variables for the ryanodine receptor (calcium release channel).
    
    Model Parameters
    ----------------
    ko : float
        Extracellular potassium concentration (mM).
    cao : float
        Extracellular calcium concentration (mM).
    nao : float
        Extracellular sodium concentration (mM).
    Vc : float
        Cytoplasmic volume (μL).
    Vsr : float
        Sarcoplasmic reticulum volume (μL).
    Vss : float
        Subsarcolemmal space volume (μL).
    R, T, F : float
        Universal gas constant, absolute temperature, and Faraday constant.
    RTONF : float
        Precomputed RT/F value for Nernst equation.
    CAPACITANCE : float
        Membrane capacitance per unit area (μF/cm²).
    gna, gcal, gkr, gks, gk1, gto : float
        Conductances for major ionic channels.
    gbna, gbca : float
        Background sodium and calcium conductances.
    gpca, gpk : float
        Pump-related conductances.
    knak, knaca : float
        Maximal Na⁺/K⁺ pump and Na⁺/Ca²⁺ exchanger rates.
    Km*, Kbuf*, Vmaxup, Vrel, etc.
        Numerous kinetic constants for buffering, pump activity, and calcium handling.

    Paper
    -----
    ten Tusscher KH, Panfilov AV. 
    Alternans and spiral breakup in a human ventricular tissue model.
    Am J Physiol Heart Circ Physiol. 2006 Sep;291(3):H1088–H1100.
    https://doi.org/10.1152/ajpheart.00109.2006

    """

    model_name = "ten_tusscher_panfilov_2006"

    def __init__(self, memory_save=True):
        """
        Initializes the ten Tusscher–Panfilov 2006 model with default parameters.
        
        Parameters
        ----------
        memory_save : bool
            Only for consistency with cpuwave version, mlx version always uses
            memory saving mode.
        """
        super().__init__()
        self.D_model = 0.154
