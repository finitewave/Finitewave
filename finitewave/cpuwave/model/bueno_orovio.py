import math
import numpy as np

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.core.model.ionic_kernel_generator import IonicKernelGenerator

from finitewave.cpuwave.stencil.sten2D.asymmetric_stencil_2d import (
    AsymmetricStencil2D
)
from finitewave.cpuwave.stencil.sten2D.isotropic_stencil_2d import (
    IsotropicStencil2D
)
from finitewave.cpuwave.stencil.sten3D.asymmetric_stencil_3d import (
    AsymmetricStencil3D
)
from finitewave.cpuwave.stencil.sten3D.isotropic_stencil_3d import (
    IsotropicStencil3D
)

from finitewave.cpuwave.model._registry import load_ops, wrap_calc
from finitewave.cpuwave.model._kernel_builder import build_kernel


try:
    ops = load_ops("bueno_orovio")
    jit_ops = wrap_calc(ops)
except KeyError as e:
    raise ImportError(
        "Bueno-Orovio model ops not found. "
        # "Install model package: pip install finitewave-model-bueno-orovio"
    ) from e


class BuenoOrovioKernel(IonicKernelGenerator):
    def __init__(self):
        super().__init__()
        self.args_order = [
            "u", "v", "w", "s", "u_o", "u_u", "theta_v", "theta_w", "theta_v_m", "theta_o",
            "tau_v1_m", "tau_v2_m", "tau_v_p", "tau_w1_m", "tau_w2_m", "k_w_m", "u_w_m",
            "tau_w_p", "tau_fi", "tau_o1", "tau_o2", "tau_so1", "tau_so2", "k_so", "u_so",
            "tau_s1", "tau_s2", "k_s", "u_s", "tau_si", "tau_w_inf", "w_inf_"
        ]

    def generate_body(self) -> str:
        model = {var: self._indexing(var) for var in (self.arrays + self.scalars)}
        u_new = f"u_new{self._raw_indexing()}"
        
        return f"""\
        J_fi = calc_Jfi({model['u']}, {model['v']}, {model['theta_v']}, 
                            {model['u_u']}, {model['tau_fi']})
        
        tau_o = calc_tau_o({model['u']}, {model['tau_o1']}, {model['tau_o2']}, {model['theta_o']})
        tau_so = calc_tau_so({model['u']}, {model['tau_so1']}, {model['tau_so2']},
                                    {model['k_so']}, {model['u_so']})
        J_so = calc_Jso({model['u']}, {model['u_o']}, {model['theta_w']},
                            tau_o, tau_so)

        J_si = calc_Jsi({model['u']}, {model['w']}, {model['s']}, {model['theta_w']}, {model['tau_si']})
        

        {u_new} += dt*calc_rhs(J_fi, J_so, J_si)

        v_inf = calc_v_inf({model['u']}, {model['theta_v_m']})
        tau_v_m = calc_tau_v_m({model['u']}, {model['theta_v_m']}, 
                                   {model['tau_v1_m']}, {model['tau_v2_m']})
        {model['v']} += dt*calc_v({model['v']}, {model['u']}, {model['theta_v']}, v_inf, 
                         tau_v_m, {model['tau_v_p']})
        
        w_inf = calc_w_inf({model['u']}, {model['theta_o']}, {model['tau_w_inf']}, {model['w_inf_']})
        tau_w_m = calc_tau_w_m({model['u']}, {model['tau_w1_m']}, {model['tau_w2_m']}, 
                                   {model['k_w_m']}, {model['u_w_m']})
        {model['w']} += dt*calc_w({model['w']}, {model['u']}, {model['theta_w']}, 
                         w_inf, tau_w_m, {model['tau_w_p']})
        
        tau_s = calc_tau_s({model['u']}, {model['tau_s1']}, {model['tau_s2']}, {model['theta_w']})
        {model['s']} += dt*calc_s({model['s']}, {model['u']}, tau_s, 
                         {model['k_s']}, {model['u_s']})

"""


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
        self.D_model = 1.
        self.npfloat    = 'float64'

        self._initialize_variables_and_parameters(ops)

    def initialize(self):
        """
        Initializes the model for simulation.
        """
        super().initialize()

        self._allocate_state_arrays()

        gen = self._initialize_kernel(BuenoOrovioKernel)

        glb = {
            "calc_v_inf": jit_ops["calc_v_inf"],
            "calc_tau_v_m": jit_ops["calc_tau_v_m"],
            "calc_v": jit_ops["calc_v"],
            "calc_w_inf": jit_ops["calc_w_inf"],
            "calc_tau_w_m": jit_ops["calc_tau_w_m"],
            "calc_w": jit_ops["calc_w"],
            "calc_tau_s": jit_ops["calc_tau_s"],
            "calc_s": jit_ops["calc_s"],
            "calc_Jfi": jit_ops["calc_Jfi"],
            "calc_tau_o": jit_ops["calc_tau_o"],
            "calc_tau_so": jit_ops["calc_tau_so"],
            "calc_Jso": jit_ops["calc_Jso"],
            "calc_Jsi": jit_ops["calc_Jsi"],
            "calc_rhs": jit_ops["calc_rhs"]
        }

        self._kernel, _ = build_kernel(
            gen=gen,
            glb=glb,
            dimensions=self.cardiac_tissue.dimensions,
            observers=self.observers,
        )

        self._buffs = self._form_and_verify_observers()
    
    def run_ionic_kernel(self):
        """
        Executes the ionic kernel for the Bueno-Orovio model.
        """
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
        """
        Selects the appropriate stencil for diffusion based on the tissue
        properties. If the tissue has fiber directions, an asymmetric stencil
        is used; otherwise, an isotropic stencil is used.

        Parameters
        ----------
        cardiac_tissue : CardiacTissue
            A tissue object representing the cardiac tissue.

        Returns
        -------
        Stencil
            The stencil object to use for diffusion computations.
        """
        if cardiac_tissue.fibers is None:
            if cardiac_tissue.dimensions == 2:
                return IsotropicStencil2D()
            elif cardiac_tissue.dimensions == 3:
                return IsotropicStencil3D()
            else:
                raise ValueError("Unsupported number of dimensions")
        else:
            if cardiac_tissue.dimensions == 2:
                return AsymmetricStencil2D()
            elif cardiac_tissue.dimensions == 3:
                return AsymmetricStencil3D()
            else:
                raise ValueError("Unsupported number of dimensions")


