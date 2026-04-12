
from finitewave.cpuwave.model._cardiac_model import CardiacModel

from finitewave.cpuwave.model._registry import load_ops, wrap_calc


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

    Model Diffusion
    ---------------
    D_model : float
        Model-specific diffusion coefficient for tissue simulations.

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
        self._initialize_variables_and_parameters(ops)
        self._initialize_model_func(ops, jit_ops)
