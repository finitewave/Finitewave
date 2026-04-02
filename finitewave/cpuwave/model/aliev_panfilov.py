import textwrap
import numpy as np

from finitewave.core.model.cardiac_model_base import CardiacModelBase
from finitewave.cpuwave.model._cardiac_model import CardiacModel

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

    def __init__(self, memory_save=False):
        """
        Initializes the AlievPanfilov instance with default parameters.
        """
        super().__init__(memory_save)
        self.D_model = 1.
        self.npfloat = 'float64'
        self._initialize_variables_and_parameters(ops)
        self._initialize_model_func(jit_ops)

    def initialize(self, simulation):
        """
        Initializes the model for simulation.
        """
        super().initialize(simulation)

        self._initialize_ionic_kernel(ops.ionic_step, self.model_func)
        self.ionic_kernel_args = [getattr(self, name) for name in self.ionic_kernel_arg_names]
        
    def run(self, dt):
        """
        Executes the ionic kernel for the Aliev-Panfilov model.
        """
        self.counter += 1
        if (self.counter - 1) % self.step != 0:
            return

        self.ionic_kernel(
            self.rhs,
            self.u,
            self.myo_indexes,
            dt,
            *self.ionic_kernel_args,
        )

    def _initialize_model_func(self, jit_ops):
        """TODO: if jit_ops func is independent each other, we can directly use jit_ops"""
        self.model_func = {
            "calc_dv": jit_ops["calc_dv"],
            "calc_rhs": jit_ops["calc_rhs"],
        }
