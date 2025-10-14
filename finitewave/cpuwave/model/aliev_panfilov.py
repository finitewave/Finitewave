import numpy as np
import jax
from numba import njit, prange

from .cardiac_model import CardiacModel
from ._registry import load_ops
from ._jitwrap import wrap_calc

ops = load_ops("aliev_panfilov")
jit_ops = wrap_calc(ops)

calc_dv = jit_ops["calc_dv"]
calc_rhs = jit_ops["calc_rhs"]


class AlievPanfilov(CardiacModel):
    """
    Implementation of the Aliev–Panfilov model of cardiac excitation for
    regular grids.

    The Aliev–Panfilov model is a phenomenological two-variable model designed to
    reproduce basic features of cardiac excitation, including wave propagation and
    reentry, while remaining computationally efficient. It uses a single recovery
    variable coupled with a cubic nonlinearity to simulate action potential dynamics
    in excitable media.

    Attributes
    ----------
    state_vars : list of str
        Names of the state variables to be saved and restored.
    memory_save : bool
        Whether to save memory by only storing the indexes of the tissue
        (``mesh > 0``).
    D_model : float
        Diffusion coefficient used for simulating spatial propagation.
    u : np.ndarray
        Transmembrane potential (dimensionless, normalized to [0, 1]).
    v : np.ndarray
        Recovery variable describing refractoriness.

    Model Parameters
    ----------------
    a : float
        Excitability threshold parameter.
    k : float
        Strength of the nonlinear source term (governs spike shape).
    eap : float
        Baseline recovery rate.
    mu_1 : float
        Recovery rate coefficient (scales v feedback).
    mu_2 : float
        Recovery rate offset (modulates u-dependence of recovery).

    Source Model
    ---------------
    https://github.com/finitewave/Aliev-Panfilov-finitewave-model/

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
        Initializes the AlievPanfilovGrid instance with default parameters.

        Parameters
        ----------
        memory_save : bool
            Whether to save memory by only storing the indexes of the tissue
            (``mesh > 0``).
        """
        super().__init__(memory_save)
        self.state_vars = ["u", "v"]
        self.D_model = 1.

        # model parameters
        parameters = ops.get_parameters()
        self.a = parameters["a"]
        self.k = parameters["k"]
        self.eps = parameters["eps"]
        self.mu1 = parameters["mu1"]
        self.mu2 = parameters["mu2"]

        # initial conditions
        variables = ops.get_variables()
        for var, val in variables.items():
            setattr(self, "init_" + var, val)

    def run(self, dt):
        """
        Executes the ionic kernel for the Aliev-Panfilov model.
        """
        ionic_kernel(self.u, self.rhs, self.myo_indexes, dt, self.v, self.a,
                     self.k, self.eps, self.mu1, self.mu2)


@njit(parallel=True)
def ionic_kernel(u, rhs, indexes, dt, v, a, k, eps, mu1, mu2):
    """
    Computes the ionic kernel for the Aliev-Panfilov 2D model.

    Parameters
    ----------
    u : np.ndarray
        Current action potential array.
    rhs : np.ndarray
        Array to store the updated action potential values.
    diffusion_indexes : np.ndarray
        Array of myocyte indices corresponding to diffusion model arrays
        (``u``, ``rhs``).
    reaction_indexes : np.ndarray
        Array of myocyte indices corresponding to cardiac model arrays
        (``v``).
    dt : float
        Time step for the simulation.
    v : np.ndarray
        Recovery variable array.
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
    """
    for i in prange(len(indexes)):
        ii = indexes[i]
        v.flat[ii] += dt * calc_dv(v.flat[ii], u.flat[ii], a, k, eps, mu1, mu2)
        rhs.flat[ii] = dt * calc_rhs(u.flat[ii], v.flat[ii], a, k)


# @jax.jit
# def ionic_kernel(u, indexes, dt, v, a, k, eps, mu1, mu2):
#     """
#     Computes the ionic kernel for the Aliev-Panfilov 2D model.

#     Parameters
#     ----------
#     u : np.ndarray
#         Current action potential array.
#     indexes : np.ndarray
#         Array of myocyte indexes corresponding to cardiac model arrays
#     dt : float
#         Time step for the simulation.
#     v : np.ndarray
#         Recovery variable array.
#     a : float
#         Excitability threshold parameter.
#     k : float
#         Strength of the nonlinear source term (governs spike shape).
#     eps : float
#         Baseline recovery rate.
#     mu1 : float
#         Recovery rate coefficient (scales v feedback).
#     mu2 : float
#         Recovery rate offset (modulates u-dependence of recovery).
#     """
#     v += dt * calc_dv(v, u, a, k, eps, mu1, mu2)
#     rhs = dt * calc_rhs(u, v, a, k)
#     return v, rhs
