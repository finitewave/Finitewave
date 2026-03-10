import numpy as np
from numba import njit, prange, typed

from .cardiac_model import CardiacModel
from ._registry import load_ops
from ._jitwrap import wrap_calc

ops = load_ops("luo_rudy_91")
jit_ops = wrap_calc(ops)

calc_rhs = jit_ops["calc_rhs"]
calc_dm = jit_ops["calc_dm"]
calc_dh = jit_ops["calc_dh"]
calc_dj = jit_ops["calc_dj"]
calc_dd = jit_ops["calc_dd"]
calc_df = jit_ops["calc_df"]
calc_dx = jit_ops["calc_dx"]
calc_dcai = jit_ops["calc_dcai"]
calc_ina = jit_ops["calc_ina"]
calc_isk = jit_ops["calc_isk"]
calc_ik = jit_ops["calc_ik"]
calc_ik1 = jit_ops["calc_ik1"]
calc_ikp = jit_ops["calc_ikp"]
calc_ib = jit_ops["calc_ib"]


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
    state_vars : list of str
        List of state variable names to save and restore (`u`, `m`, `h`, `j`, `d`, `f`, `x`, `cai`).
    D_model : float
        Diffusion coefficient representing electrical conductivity in the medium (typically set to 0.1).
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

    def __init__(self, memory_save=False):
        """
        Initializes the LuoRudy91 instance, setting up the state variables and parameters.
        """
        super().__init__(memory_save)
        self.D_model = 0.1

        self.state_vars = ["u", "m", "h", "j", "d", "f", "x", "cai"]
        self.npfloat = 'float64'

        # initial conditions
        params = ops.get_parameters()
        for var, val in params.items():
            setattr(self, var, val)

        # initial conditions
        variables = ops.get_variables()
        for var, val in variables.items():
            setattr(self, "init_" + var, val)

    def run(self, dt):
        """
        Executes the ionic kernel to update the state variables and membrane
        potential.
        """
        self.counter += 1
        if (self.counter - 1) % self.step != 0:
            return

        ionic_kernel(self.u, self.rhs, self.myo_indexes, dt,
                     self.m, self.h, self.j, self.d, self.f, self.x, self.cai,
                     self.gna, self.gsi, self.gk, self.gk1, self.gkp, self.gb, 
                     self.ko, self.ki, self.nai, self.nao, self.R, self.T, 
                     self.F, self.PR_NaK, self.E_Na, self.E_K1)
        
    def prepacing(self, stim_sequence):
        stim_values = []
        t_max = 0

        for stim in stim_sequence:
            n_beats = stim["n_beats"]
            dt = stim["dt"]
            bcl = stim["cycle_length"]
            duration = stim["stim_duration"]
            stim_amplitude = stim["stim_amplitude"]

            stim_val = self.build_prepacing(dt, n_beats, bcl, duration, stim_amplitude)
            stim_values.append(stim_val)
            t_max += dt * len(stim_val)

        stim_values = np.concatenate(stim_values)
        self.u_pacing, state_vars = prepacing(
            dt, t_max, stim_values, self.init_u,
            self.init_m, self.init_h, self.init_j, self.init_d, self.init_f,
            self.init_x, self.init_cai,
            self.gna, self.gsi, self.gk, self.gk1, self.gkp, self.gb, 
            self.ko, self.ki, self.nai, self.nao, self.R, self.T, 
            self.F, self.PR_NaK, self.E_Na, self.E_K1)
        # print(state_vars)
        # initial conditions
        for var, val in state_vars.items():
            setattr(self, "init_" + var, val)


@njit(parallel=True)
def ionic_kernel(u, rhs, indexes, dt,
                 m, h, j, d, f, x, cai,
                 gna, gsi, gk, gk1, gkp, gb, ko, ki, nai, nao, R, T, F, PR_NaK,
                 E_Na, E_K1):
    """
    Computes the ionic currents and updates the state variables in the 2D
    Luo-Rudy 1991 cardiac model.

    Parameters
    ----------
    u : np.ndarray
        Array of the current membrane potential values.
    rhs : np.ndarray
        Array to store the computed ionic currents
    indexes : np.ndarray
        Array of indexes where the kernel should be computed (``mesh == 1``).
    dt : float
        Time step for the simulation.
    m : np.ndarray
        Array for the gating variable `m`.
    h : np.ndarray
        Array for the gating variable `h`.
    j : np.ndarray
        Array for the gating variable `j`.
    d : np.ndarray
        Array for the gating variable `d`.
    f : np.ndarray
        Array for the gating variable `f`.
    x : np.ndarray
        Array for the gating variable `x`.
    cai : np.ndarray
        Array for the intracellular calcium concentration.
    gna : float
        Maximum sodium conductance [mS/μF].
    gsi : float
        Maximum calcium conductance [mS/μF].
    gk : float
        Maximum potassium conductance [mS/μF].
    gk1 : float
        Maximum inward rectifier potassium conductance [mS/μF].
    gkp : float
        Maximum plateau potassium conductance [mS/μF].
    gb : float
        Background conductance [mS/μF].
    ko : float
        Extracellular potassium concentration [mM].
    ki : float
        Intracellular potassium concentration [mM].
    nai : float
        Intracellular sodium concentration [mM].
    nao : float
        Extracellular sodium concentration [mM].
    R : float
        Universal gas constant (J/(mol·K)).
    T : float
        Temperature (Kelvin).
    F : float
        Faraday constant (C/mmol).
    PR_NaK : float
        Sodium/potassium permeability ratio (used in reversal potential calculation for I_K).
    """
    for ind in prange(len(indexes)):
        ii = indexes[ind]

        # Fast sodium current:
        m.flat[ii] += dt * calc_dm(u.flat[ii], m.flat[ii])
        h.flat[ii] += dt * calc_dh(u.flat[ii], h.flat[ii])
        j.flat[ii] += dt * calc_dj(u.flat[ii], j.flat[ii])

        ina = calc_ina(u.flat[ii], m.flat[ii], h.flat[ii], j.flat[ii], E_Na, gna)

        # Slow inward current:
        d.flat[ii] += dt * calc_dd(u.flat[ii], d.flat[ii])
        f.flat[ii] += dt * calc_df(u.flat[ii], f.flat[ii])

        isi = calc_isk(u.flat[ii], d.flat[ii], f.flat[ii], cai.flat[ii], gsi)

        cai.flat[ii] += dt * calc_dcai(cai.flat[ii], isi)

        # Time-dependent potassium current:
        x.flat[ii] += dt * calc_dx(u.flat[ii], x.flat[ii])
        # Time-dependent potassium current:
        ik = calc_ik(u.flat[ii], x.flat[ii], ko, ki, nao, nai, PR_NaK, R, T, F, gk)

        # Time-independent potassium current:
        ik1 = calc_ik1(u.flat[ii], ko, E_K1, gk1)

        # Plateau potassium current:
        ikp = calc_ikp(u.flat[ii], E_K1, gkp)

        # Background current:
        ib = calc_ib(u.flat[ii], gb)

        rhs.flat[ii] = calc_rhs(ina, isi, ik, ik1, ikp, ib)


@njit
def prepacing(dt, t_max, stim_values, u,
                 m, h, j, d, f, x, cai,
                 gna, gsi, gk, gk1, gkp, gb, ko, ki, nai, nao, R, T, F, PR_NaK,
                 E_Na, E_K1):
    u_list = np.zeros((int(t_max/dt),), dtype=np.float64)
    u_list[0] = u
    
    for i in range(1, int(t_max/dt)):

        u += stim_values[i]

        # Fast sodium current:
        m += dt * calc_dm(u, m)
        h += dt * calc_dh(u, h)
        j += dt * calc_dj(u, j)

        ina = calc_ina(u, m, h, j, E_Na, gna)

        # Slow inward current:
        d += dt * calc_dd(u, d)
        f += dt * calc_df(u, f)

        isi = calc_isk(u, d, f, cai, gsi)

        cai += dt * calc_dcai(cai, isi)

        # Time-dependent potassium current:
        x += dt * calc_dx(u, x)
        # Time-dependent potassium current:
        ik = calc_ik(u, x, ko, ki, nao, nai, PR_NaK, R, T, F, gk)

        # Time-independent potassium current:
        ik1 = calc_ik1(u, ko, E_K1, gk1)

        # Plateau potassium current:
        ikp = calc_ikp(u, E_K1, gkp)

        # Background current:
        ib = calc_ib(u, gb)

        rhs = calc_rhs(ina, isi, ik, ik1, ikp, ib)
        u = u + dt * rhs
        u_list[i] = u

    # m, h, j, d, f, x, cai

    state_vars = typed.Dict()
    state_vars['u'] = u
    state_vars['m'] = m
    state_vars['h'] = h
    state_vars['j'] = j
    state_vars['d'] = d
    state_vars['f'] = f
    state_vars['x'] = x
    state_vars['cai'] = cai

    return u_list, state_vars
