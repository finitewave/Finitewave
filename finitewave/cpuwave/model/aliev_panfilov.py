import numpy as np
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
        self.counter += 1
        if (self.counter - 1) % self.step != 0:
            return

        ionic_kernel(self.u, self.rhs, self.myo_indexes, dt, self.v, self.a,
                     self.k, self.eps, self.mu1, self.mu2)
        
    def prepacing(self, stim_sequence):
        stim_values = []
        t_max = 0

        for stim in stim_sequence:
            n_beats = stim["n_beats"]
            dt = stim["dt"]
            bcl = stim["cycle_length"]
            duration = stim["stim_duration"]
            stim_amplitude = stim["stim_amplitude"]

            stim_val = self._build_prepacing(dt, n_beats, bcl, duration, stim_amplitude)
            stim_values.append(stim_val)
            t_max += dt * len(stim_val)

        stim_values = np.concatenate(stim_values)
        self.u_pacing, state_vars = prepacing(
            dt, t_max, stim_values, self.init_u, self.init_v, self.a,
            self.k, self.eps, self.mu1, self.mu2)
        
        # print(state_vars)
        # initial conditions
        for var, val in state_vars.items():
            if var == "j":
                var += "_"
            setattr(self, "init_" + var, val)


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
        rhs.flat[ii] = calc_rhs(u.flat[ii], v.flat[ii], a, k)


@njit
def prepacing(dt, t_max, stim_values, u, v, a, k, eps, mu1, mu2):
    """
    Computes the ionic kernel for the Aliev-Panfilov 2D model.

    Parameters
    ----------
    dt : float
        Time step for the simulation.
    t_max : float
        Total time for the pre-pacing simulation.
    stim_values : np.ndarray
        Array of stimulus values to be applied at each time step.
    u : float
        Initial action potential value.
    v : float
        Initial recovery variable value.
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
    u_list = np.zeros((int(t_max/dt),), dtype=np.float64)
    u_list[0] = u

    for i in range(1, int(t_max/dt)):
        u += stim_values[i]

        v += dt * calc_dv(v, u, a, k, eps, mu1, mu2)
        rhs = calc_rhs(u, v, a, k)

        u = u + dt * rhs
        u_list[i] = u

    state_vars = typed.Dict()
    state_vars['u'] = u
    state_vars['v'] = v

    return u_list, state_vars
