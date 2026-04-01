import textwrap
import numpy as np

from finitewave.core.model.cardiac_model_base import CardiacModelBase
from .kernel_generators import (
    StepKernelGenerator,
    SingleCellKernelGenerator
)

from finitewave.cpuwave.model._registry import load_ops, wrap_calc
from finitewave.cpuwave.model._kernel_builder import build_kernel


try:
    ops = load_ops("aliev_panfilov")
    jit_ops = wrap_calc(ops)
except KeyError as e:
    raise ImportError(
        "Aliev-Panfilov model ops not found. "
        "Install model package: pip install finitewave-model-aliev-panfilov"
    ) from e


class AlievPanfilovStepKernel(StepKernelGenerator):
    def __init__(self):
        super().__init__()
        self.args_order = [
            "u", "v", "a", "k", "mu1", "mu2", "eps"
        ]
        self.state_vars = ["u", "v"]

    def generate_body(self) -> str:
        asign_vars = "\n".join(f"{var}_old = {self._indexing(var)}" for var in self.arrays)
        step_body = "\n" + self.extract_func_body(ops.step)
        update_vars = "\n".join(f"{var}_new = {var}_old" for var in self.state_vars)

        body = f"""\
            {asign_vars}
            {step_body}
            {update_vars}
        """
        return textwrap.dedent(body).strip()
    
class AlievPanfilovPrepacingKernel(PrepacingKernelGenerator):
    def __init__(self):
        super().__init__()
        self.kernel_func_name = "prepacing_kernel"
        self.arrays = ["u"]
        self.scalars = ["a", "k", "eps", "mu1", "mu2"]
        self.common_args = ["rhs", "indexes", "dt", "step"]
        self.model_args = self.arrays + self.scalars
        self.observers = []


class AlievPanfilov(CardiacModel):
    """
    Implementation of the Aliev–Panfilov model of cardiac excitation.

    The Aliev–Panfilov model is a phenomenological two-variable model designed to
    reproduce basic features of cardiac excitation, including wave propagation and
    reentry, while remaining computationally efficient. It uses a single recovery
    variable coupled with a cubic nonlinearity to simulate action potential dynamics
    in excitable media.

    Attributes
    ----------
    D_model : float
        Diffusion coefficient used for simulating spatial propagation.
    npfloat : str
        Floating-point precision used in the simulation (default: 'float64').

    Model Variables
    ---------------
    u : np.ndarray
        Transmembrane potential (dimensionless, normalized to [0,1]).
    v : np.ndarray
        Recovery variable describing refractoriness.

    Model Parameters
    ----------------
    a : float
        Excitability threshold parameter.
    k : float
        Strength of the nonlinear source term (governs spike shape).
    eps : float
        Baseline recovery rate.
    mu1 : float
        Recovery rate coefficient (scales v feedback).
    mu2 : float
        Recovery rate offset (modulates u-dependence of recovery).

    Paper
    -----
    Rubin R. Aliev, Alexander V. Panfilov,
    A simple two-variable model of cardiac excitation,
    Chaos, Solitons & Fractals,
    Volume 7, Issue 3,
    1996,
    Pages 293-301,
    ISSN 0960-0779,
    https://doi.org/10.1016/0960-0779(95)00089-5.
    """

    def __init__(self):
        """
        Initializes the AlievPanfilov instance with default parameters.
        """
        super().__init__()
        self.D_model = 1.
        self.npfloat = 'float64'

        self._initialize_variables_and_parameters(ops)

    def initialize(self):
        """
        Initializes the model for simulation.
        """
        super().initialize()

        self._allocate_state_arrays()

        gen = self._initialize_kernel(AlievPanfilovKernel)

        glb = {
            "calc_dv": jit_ops["calc_dv"], 
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
        Executes the ionic kernel for the Aliev-Panfilov model.
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
