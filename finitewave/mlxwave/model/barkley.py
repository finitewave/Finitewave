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
    ops = load_ops("barkley")
    jit_ops = wrap_calc(ops)
except KeyError as e:
    raise ImportError(
        "Barkley model ops not found. "
        "Install model package: pip install finitewave-model-barkley"
    ) from e


class BarkleyKernel(IonicKernelGenerator):
    def __init__(self):
        super().__init__()
        self.args_order = [
            "u", "v", "a", "b", "eps"
        ]

    def generate_body(self) -> str:
        model = {var: self._indexing(var) for var in (self.arrays + self.scalars)}
        u_new = f"u_new{self._raw_indexing()}"

        return f"""\
        {u_new} += dt * calc_rhs({model['u']}, {model['v']}, {model['a']}, 
            {model['b']}, {model['eps']})

        {model['v']} += dt * calc_dv({model['v']}, {model['u']})
"""


class Barkley(CardiacModel):
    """
    Implementation of the Barkley model for excitable media.

    The Barkley model is a simplified two-variable reaction–diffusion system
    originally developed to study wave propagation in excitable media. While it is 
    not biophysically detailed, it captures essential qualitative features of 
    cardiac-like excitation dynamics such as spiral waves, wave break, and reentry.

    This implementation is included for benchmarking, educational purposes, 
    and comparison against more detailed cardiac models.

    Attributes
    ----------
    D_model : float
        Diffusion coefficient for excitation variable.
    npfloat : str
        Floating-point precision (default: 'float64').

    Model Variables
    ---------------
    u : np.ndarray
        Excitation variable (analogous to membrane potential).
    v : np.ndarray
        Recovery variable controlling excitability.
    
    Model Parameters
    ----------------
    a : float
        Threshold-like parameter controlling excitability.
    b : float
        Recovery time scale.
    eap : float
        Controls sharpness of the activation term (nonlinear gain).

    Paper
    -----
    Barkley, D. (1991).
    A model for fast computer simulation of waves in excitable media.
    Physica D: Nonlinear Phenomena, 61-70.
    https://doi.org/10.1016/0167-2789(86)90198-1.

    """
    def __init__(self):
        super().__init__()
        self.D_model = 1.0
        self.npfloat = "float64"

        self._initialize_variables_and_parameters(ops)

    def initialize(self):
        super().initialize()

        self._allocate_state_arrays()

        gen = self._initialize_kernel(BarkleyKernel)
    
        glb = {
            "calc_dv": jit_ops["calc_dv"],
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


