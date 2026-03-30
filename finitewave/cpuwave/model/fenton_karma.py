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
    ops = load_ops("fenton_karma")
    jit_ops = wrap_calc(ops)
except KeyError as e:
    raise ImportError(
        "Fenton-Karma model ops not found. "
        # "Install model package: pip install finitewave-model-fenton-karma"
    ) from e


class FentonKarmaKernel(IonicKernelGenerator):
    def __init__(self):
        super().__init__()
        self.args_order = [
            "u", "v", "w", "tau_d", "tau_o", "tau_r", "tau_si", "tau_v_m", "tau_v_p",
            "tau_w_m", "tau_w_p", "k", "u_c", "uc_si"
        ]

    def generate_body(self) -> str:
        model = {var: self._indexing(var) for var in (self.arrays + self.scalars)}
        u_new = f"u_new{self._raw_indexing()}"
        
        return f"""\
        J_fi = calc_Jfi({model['u']}, {model['v']}, 
                            {model['u_c']}, {model['tau_d']})
        J_so = calc_Jso({model['u']}, {model['u_c']},
                            {model['tau_o']}, {model['tau_r']})
        J_si = calc_Jsi({model['u']}, {model['w']},
                            {model['k']}, {model['uc_si']}, {model['tau_si']})

        {u_new} += dt * calc_rhs(J_fi, J_so, J_si)

        {model['v']} += dt * calc_dv({model['v']} , {model['u']} , 
                                          {model['u_c']} , {model['tau_v_m']} , {model['tau_v_p']} )
        {model['w']}  += dt * calc_dw({model['w']} , {model['u']} , 
                                          {model['u_c']} , {model['tau_w_m']} , {model['tau_w_p']})

"""


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

    def __init__(self):
        """
        Initializes the Fenton-Karma instance with default parameters.
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

        gen = self._initialize_kernel(FentonKarmaKernel)
        
        glb = {
            "calc_dv": jit_ops["calc_dv"], 
            "calc_dw": jit_ops["calc_dw"],
            "calc_Jfi": jit_ops["calc_Jfi"],
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
        Executes the ionic kernel for the Fenton-Karma model.
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


