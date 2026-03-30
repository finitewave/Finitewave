import math
import numpy as np

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.core.model.ionic_kernel_generator import IonicKernelGenerator

from finitewave.cpuwave.stencil.sten2D.asymmetric_stencil_2d import AsymmetricStencil2D
from finitewave.cpuwave.stencil.sten2D.isotropic_stencil_2d import IsotropicStencil2D
from finitewave.cpuwave.stencil.sten3D.asymmetric_stencil_3d import AsymmetricStencil3D
from finitewave.cpuwave.stencil.sten3D.isotropic_stencil_3d import IsotropicStencil3D

from finitewave.cpuwave.model._registry import load_ops, wrap_calc
from finitewave.cpuwave.model._kernel_builder import build_kernel


try:
    ops = load_ops("luo_rudy_91")
    jit_ops = wrap_calc(ops)
except KeyError as e:
    raise ImportError(
        "Luo–Rudy 1991 model ops not found."
        # "Install model package: pip install finitewave-model-luo-rudy91"
    ) from e


class LuoRudy91Kernel(IonicKernelGenerator):
    def __init__(self):
        super().__init__()
        self.args_order = [
            "u", "m", "h", "j", "d", "f", "x", "cai",
            "gna", "gsi", "gk", "gk1", "gkp", "gb",
            "ko", "ki", "nai", "nao", "cao",
            "R", "T", "F", "E_Na", "E_K1", "PR_NaK"
        ]

    def generate_body(self) -> str:
        model = {var: self._indexing(var) for var in (self.arrays + self.scalars)}
        u_new = f"u_new{self._raw_indexing()}"

        return f"""\
        u_loc = {model['u']}

        ina = calc_ina(u_loc, {model['m']}, {model['h']}, {model['j']}, {model['E_Na']}, {model['gna']})
        isi = calc_isk(u_loc, {model['d']}, {model['f']}, {model['cai']}, {model['gsi']})
        {model['cai']} += dt * calc_dcai({model['cai']}, isi)

        ik  = calc_ik(u_loc, {model['x']}, {model['ko']}, {model['ki']}, {model['nao']}, {model['nai']}, 
            {model['PR_NaK']}, {model['R']}, {model['T']}, {model['F']}, {model['gk']})
        ik1 = calc_ik1(u_loc, {model['ko']}, {model['E_K1']}, {model['gk1']})
        ikp = calc_ikp(u_loc, {model['E_K1']}, {model['gkp']})
        ib  = calc_ib(u_loc, {model['gb']})

        {u_new} += dt * calc_rhs(ina, isi, ik, ik1, ikp, ib)

        {model['m']} += dt * calc_dm(u_loc, {model['m']})
        {model['h']} += dt * calc_dh(u_loc, {model['h']})
        {model['j']} += dt * calc_dj(u_loc, {model['j']})

        {model['d']} += dt * calc_dd(u_loc, {model['d']})
        {model['f']} += dt * calc_df(u_loc, {model['f']})
        {model['x']} += dt * calc_dx(u_loc, {model['x']})
    """



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
    def __init__(self):
        super().__init__()

        self.D_model = 0.1
        self.npfloat = "float64"

        self._initialize_variables_and_parameters(ops)

    def initialize(self):
        super().initialize()
    
        self._allocate_state_arrays()
    
        gen = self._initialize_kernel(LuoRudy91Kernel)
    
        glb = {
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

        self._kernel, _ = build_kernel(
            gen=gen,
            glb=glb,
            dimensions=self.cardiac_tissue.dimensions,
            observers=self.observers,
        )
        self._buffs = self._form_and_verify_observers()

    def run_ionic_kernel(self):
        args = [getattr(self, name) for name in self._kernel_args_order]
        self._kernel(
            self.u_new,
            self.cardiac_tissue.myo_indexes,
            self.dt,
            self.step,
            *args,
            *self._buffs,
        )

    def select_stencil(self, cardiac_tissue):
        if cardiac_tissue.fibers is None:
            if cardiac_tissue.dimensions == 2:
                return IsotropicStencil2D()
            if cardiac_tissue.dimensions == 3:
                return IsotropicStencil3D()
            raise ValueError("Unsupported number of dimensions")
        else:
            if cardiac_tissue.dimensions == 2:
                return AsymmetricStencil2D()
            if cardiac_tissue.dimensions == 3:
                return AsymmetricStencil3D()
            raise ValueError("Unsupported number of dimensions")
