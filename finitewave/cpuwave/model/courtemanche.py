import numpy as np
from numba import njit, prange, typed

from .cardiac_model import CardiacModel
from ._registry import load_ops
from ._jitwrap import wrap_calc

ops = load_ops("courtemanche")
jit_ops = wrap_calc(ops)

calc_rhs = jit_ops["calc_rhs"]
calc_where = jit_ops["calc_where"]
calc_gating_variable = jit_ops["calc_gating_variable"]
calc_gating_variable_rush_larsen = jit_ops["calc_gating_variable_rush_larsen"]
calc_cmdn = jit_ops["calc_cmdn"]
calc_trpn = jit_ops["calc_trpn"]
calc_csqn = jit_ops["calc_csqn"]
calc_dnai = jit_ops["calc_dnai"]
calc_dki = jit_ops["calc_dki"]
calc_dcai = jit_ops["calc_dcai"]
calc_dcaup = jit_ops["calc_dcaup"]
calc_dcarel = jit_ops["calc_dcarel"]
calc_equilibrum_potentials = jit_ops["calc_equilibrum_potentials"]
calc_ina = jit_ops["calc_ina"]
calc_gating_m = jit_ops["calc_gating_m"]
calc_gating_h = jit_ops["calc_gating_h"]
calc_gating_j = jit_ops["calc_gating_j"]
calc_ik1 = jit_ops["calc_ik1"]
calc_ito = jit_ops["calc_ito"]
calc_ikur = jit_ops["calc_ikur"]
calc_ikr = jit_ops["calc_ikr"]
calc_iks = jit_ops["calc_iks"]
calc_ical = jit_ops["calc_ical"]
calc_inak = jit_ops["calc_inak"]
calc_inaca = jit_ops["calc_inaca"]
calc_ibca = jit_ops["calc_ibca"]
calc_ibna = jit_ops["calc_ibna"]
calc_ipca = jit_ops["calc_ipca"]
calc_irel = jit_ops["calc_irel"]
calc_itr = jit_ops["calc_itr"]
calc_iup = jit_ops["calc_iup"]
calc_iupleak = jit_ops["calc_iupleak"]


class Courtemanche(CardiacModel):
    """
    A class to represent the Courtemanche cardiac model.

    Attributes
    ----------
    D_model : float
        Model specific diffusion coefficient.
    state_vars : list of str
        List of state variable names.
    """

    def __init__(self, memory_save=False):
        super().__init__(memory_save)
        self.D_model = 0.1544
        self.state_vars = ["u", "nai", "ki", "cai", "caup", "carel", "m", "h",
                           "j_", "d", "f", "oa", "oi", "ua", "ui", "xr", "xs",
                           "fca", "irel", "vrel", "urel", "wrel"]
        # Model parameters
        parameters = ops.get_parameters()
        self.gna = parameters["gna"]
        self.gnab = parameters["gnab"]
        self.gk1 = parameters["gk1"]
        self.gkr = parameters["gkr"]
        self.gks = parameters["gks"]
        self.gto = parameters["gto"]
        self.gcal = parameters["gcal"]
        self.gcab = parameters["gcab"]
        self.gkur_coeff = parameters["gkur_coeff"]
        self.F = parameters["F"]
        self.T = parameters["T"]
        self.R = parameters["R"]
        self.Vc = parameters["Vc"]
        self.Vj = parameters["Vj"]
        self.Vup = parameters["Vup"]
        self.Vrel = parameters["Vrel"]
        self.ibk = parameters["ibk"]
        self.cao = parameters["cao"]
        self.nao = parameters["nao"]
        self.ko = parameters["ko"]
        self.caupmax = parameters["caupmax"]
        self.kup = parameters["kup"]
        self.kmnai = parameters["kmnai"]
        self.kmko = parameters["kmko"]
        self.kmnancx = parameters["kmnancx"]
        self.kmcancx = parameters["kmcancx"]
        self.ksatncx = parameters["ksatncx"]
        self.kmcmdn = parameters["kmcmdn"]
        self.kmtrpn = parameters["kmtrpn"]
        self.kmcsqn = parameters["kmcsqn"]
        self.trpnmax = parameters["trpnmax"]
        self.cmdnmax = parameters["cmdnmax"]
        self.csqnmax = parameters["csqnmax"]
        self.inacamax = parameters["inacamax"]
        self.inakmax = parameters["inakmax"]
        self.ipcamax = parameters["ipcamax"]
        self.krel = parameters["krel"]
        self.iupmax = parameters["iupmax"]
        self.kq10 = parameters["kq10"]

        # initial conditions
        variables = ops.get_variables()
        for var, val in variables.items():
            if var == "j":
                var += "_"
            setattr(self, "init_" + var, val)

    def run(self, dt):
        """
        Executes the ionic kernel function to update ionic currents and state
        variables

        Parameters
        ----------
        u : np.ndarray
            Current action potential array.
        rhs : np.ndarray
            Array to store the updated action potential values.
        indexes : np.ndarray
            Array of myocyte indices corresponding to diffusion model arrays
            (``u``, ``rhs``).
        dt : float
            Time step for the simulation.
        """
        self.counter += 1
        if (self.counter - 1) % self.step != 0:
            return

        ionic_kernel(self.u, self.rhs, self.myo_indexes, dt,
                     self.nai, self.ki,
                     self.cai, self.caup, self.carel, self.m, self.h, self.j_,
                     self.d, self.f, self.oa, self.oi, self.ua, self.ui,
                     self.xr, self.xs, self.fca, self.irel, self.vrel,
                     self.urel, self.wrel,
                     self.gna, self.gnab, self.gk1, self.gkr, self.gks,
                     self.gto, self.gcal, self.gcab, self.gkur_coeff, self.F,
                     self.T, self.R, self.Vc, self.Vj, self.Vup, self.Vrel,
                     self.ibk, self.cao, self.nao, self.ko, self.caupmax,
                     self.kup, self.kmnai, self.kmko, self.kmnancx,
                     self.kmcancx, self.ksatncx, self.kmcmdn, self.kmtrpn,
                     self.kmcsqn, self.trpnmax, self.cmdnmax, self.csqnmax,
                     self.inacamax, self.inakmax, self.ipcamax, self.krel,
                     self.iupmax, self.kq10)
        
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
            dt, t_max, stim_values, self.init_u,
            self.init_nai, self.init_ki,
            self.init_cai, self.init_caup, self.init_carel, self.init_m, self.init_h, self.init_j_,
            self.init_d, self.init_f, self.init_oa, self.init_oi, self.init_ua, self.init_ui,
            self.init_xr, self.init_xs, self.init_fca, self.init_irel, self.init_vrel,
            self.init_urel, self.init_wrel,
            self.gna, self.gnab, self.gk1, self.gkr, self.gks,
            self.gto, self.gcal, self.gcab, self.gkur_coeff, self.F,
            self.T, self.R, self.Vc, self.Vj, self.Vup, self.Vrel,
            self.ibk, self.cao, self.nao, self.ko, self.caupmax,
            self.kup, self.kmnai, self.kmko, self.kmnancx,
            self.kmcancx, self.ksatncx, self.kmcmdn, self.kmtrpn,
            self.kmcsqn, self.trpnmax, self.cmdnmax, self.csqnmax,
            self.inacamax, self.inakmax, self.ipcamax, self.krel,
            self.iupmax, self.kq10)
        
        # print(state_vars)
        # initial conditions
        for var, val in state_vars.items():
            if var == "j":
                var += "_"
            setattr(self, "init_" + var, val)
            

    def _build_prepacing(self, dt, n_beats, bcl, stim_duration, stim_amplitude):
        t_max = n_beats * bcl

        stim_values = np.zeros(int(t_max / dt), dtype=np.float64)

        for s in np.arange(n_beats):
            stim_start = s * bcl
            stim_end = stim_start + stim_duration
            
            start_idx = int(stim_start / dt)
            end_idx = int(stim_end / dt)
            stim_values[start_idx: end_idx] = dt * stim_amplitude

        return stim_values


@njit(parallel=True, fastmath=True, cache=True)
def ionic_kernel(u, rhs, indexes, dt,
                 nai, ki, cai, caup, carel, m, h, j_, d, f, oa, oi, ua, ui, xs,
                 xr, fca, irel, vrel, urel, wrel,
                 gna, gnab, gk1, gkr, gks, gto, gcal, gcab, gkur_coeff, F, T,
                 R, Vc, Vj, Vup, Vrel, ibk, cao, nao, ko, caupmax, kup, kmnai,
                 kmko, kmnancx, kmcancx, ksatncx, kmcmdn, kmtrpn, kmcsqn,
                 trpnmax, cmdnmax, csqnmax, inacamax, inakmax, ipcamax, krel,
                 iupmax, kq10):
    """
    Computes the ionic currents and updates the state variables in the 2D
    Courtemanche cardiac model.

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
    ... : np.ndarray
        Arrays of state variables
    ... : float
        Model parameters
    """

    for i in prange(indexes.shape[0]):
        ii = indexes[i]

        ena, ek, eca = calc_equilibrum_potentials(nai.flat[ii], nao,
                                                  ki.flat[ii], ko,
                                                  cai.flat[ii], cao, R, T, F,
                                                  where=calc_where)

        m.flat[ii] = calc_gating_m(m.flat[ii], u.flat[ii], dt,
                                   where=calc_where)
        h.flat[ii] = calc_gating_h(h.flat[ii], u.flat[ii], dt,
                                   where=calc_where)
        j_.flat[ii] = calc_gating_j(j_.flat[ii], u.flat[ii], dt,
                                    where=calc_where)

        ina = calc_ina(u.flat[ii], m.flat[ii], h.flat[ii], j_.flat[ii],
                       gna, ena)
        ik1 = calc_ik1(u.flat[ii], gk1, ek)
        ito, oa.flat[ii], oi.flat[ii] = calc_ito(u.flat[ii], dt, kq10,
                                                 oa.flat[ii], oi.flat[ii],
                                                 gto, ek)
        ikur, ua.flat[ii], ui.flat[ii] = calc_ikur(u.flat[ii], dt, kq10,
                                                   ua.flat[ii], ui.flat[ii],
                                                   ek, gkur_coeff)
        ikr, xr.flat[ii] = calc_ikr(u.flat[ii], dt, xr.flat[ii], gkr, ek)
        iks, xs.flat[ii] = calc_iks(u.flat[ii], dt, xs.flat[ii], gks, ek)
        ical, d.flat[ii], f.flat[ii], fca.flat[ii] = calc_ical(u.flat[ii],
                                                               dt,
                                                               d.flat[ii],
                                                               f.flat[ii],
                                                               cai.flat[ii],
                                                               gcal,
                                                               fca.flat[ii])
        inak = calc_inak(inakmax, nai.flat[ii], nao, ko, kmnai, kmko, F,
                         u.flat[ii], R, T)
        inaca = calc_inaca(inacamax, nai.flat[ii], nao, cai.flat[ii], cao,
                           kmnancx, kmcancx, ksatncx, F, u.flat[ii], R, T)
        ibca = calc_ibca(gcab, eca, u.flat[ii])
        ibna = calc_ibna(gnab, ena, u.flat[ii])
        ipca = calc_ipca(ipcamax, cai.flat[ii])
        irel.flat[ii], urel.flat[ii], vrel.flat[ii], wrel.flat[ii] = calc_irel(
            dt, urel.flat[ii], vrel.flat[ii], irel.flat[ii], wrel.flat[ii],
            ical, inaca, krel, carel.flat[ii], cai.flat[ii], u.flat[ii],
            F, Vrel)
        itr = calc_itr(caup.flat[ii], carel.flat[ii])
        iup = calc_iup(iupmax, cai.flat[ii], kup)
        iupleak = calc_iupleak(caup.flat[ii], caupmax, iupmax)

        caup.flat[ii] += dt * calc_dcaup(iup, iupleak, itr, Vrel, Vup)
        nai.flat[ii] += dt * calc_dnai(inak, inaca, ibna, ina, F, Vj)

        ki.flat[ii] += dt * calc_dki(inak, ik1, ito, ikur, ikr, iks, ibk, F,
                                     Vj)
        cai.flat[ii] += dt * calc_dcai(cai.flat[ii], inaca, ipca, ical, ibca,
                                       iup, iupleak, irel.flat[ii], Vrel,
                                       Vup, trpnmax, kmtrpn, cmdnmax, kmcmdn,
                                       F, Vj)

        carel.flat[ii] += dt * calc_dcarel(carel.flat[ii], itr,
                                           irel.flat[ii], csqnmax, kmcsqn)

        rhs.flat[ii] = (- calc_rhs(ina, ik1, ito, ikur, ikr, iks, ical,
                                   ipca, inak, inaca, ibna, ibca))


@njit
def prepacing(dt, t_max, stim_values, u,
              nai, ki, cai, caup, carel, m, h, j_, d, f, oa, oi, ua, ui, xs,
              xr, fca, irel, vrel, urel, wrel,
              gna, gnab, gk1, gkr, gks, gto, gcal, gcab, gkur_coeff, F, T,
              R, Vc, Vj, Vup, Vrel, ibk, cao, nao, ko, caupmax, kup, kmnai,
              kmko, kmnancx, kmcancx, ksatncx, kmcmdn, kmtrpn, kmcsqn,
              trpnmax, cmdnmax, csqnmax, inacamax, inakmax, ipcamax, krel,
              iupmax, kq10):
        
    u_list = np.zeros((int(t_max/dt),), dtype=np.float64)
    u_list[0] = u
    
    for i in range(1, int(t_max/dt)):

        u += stim_values[i]

        ena, ek, eca = calc_equilibrum_potentials(nai, nao, ki, ko, cai, cao,
                                                  R, T, F, where=calc_where)

        m = calc_gating_m(m, u, dt, where=calc_where)
        h = calc_gating_h(h, u, dt, where=calc_where)
        j_ = calc_gating_j(j_, u, dt, where=calc_where)

        ina = calc_ina(u, m, h, j_, gna, ena)
        ik1 = calc_ik1(u, gk1, ek)
        ito, oa, oi = calc_ito(u, dt, kq10, oa, oi, gto, ek)
        ikur, ua, ui = calc_ikur(u, dt, kq10, ua, ui, ek, gkur_coeff)
        ikr, xr = calc_ikr(u, dt, xr, gkr, ek)
        iks, xs = calc_iks(u, dt, xs, gks, ek)
        ical, d, f, fca = calc_ical(u, dt, d, f, cai, gcal, fca)
        inak = calc_inak(inakmax, nai, nao, ko, kmnai, kmko, F, u, R, T)
        inaca = calc_inaca(inacamax, nai, nao, cai, cao, kmnancx, kmcancx,
                            ksatncx, F, u, R, T)
        ibca = calc_ibca(gcab, eca, u)
        ibna = calc_ibna(gnab, ena, u)
        ipca = calc_ipca(ipcamax, cai)
        irel, urel, vrel, wrel = calc_irel(dt, urel, vrel, irel, wrel,
                                            ical, inaca, krel, carel, cai, u,
                                            F, Vrel)
        itr = calc_itr(caup, carel)
        iup = calc_iup(iupmax, cai, kup)
        iupleak = calc_iupleak(caup, caupmax, iupmax)

        caup += dt * calc_dcaup(iup, iupleak, itr, Vrel, Vup)
        nai += dt * calc_dnai(inak, inaca, ibna, ina, F, Vj)

        ki += dt * calc_dki(inak, ik1, ito, ikur, ikr, iks, ibk, F, Vj)
        cai += dt * calc_dcai(cai, inaca, ipca, ical, ibca, iup, iupleak,
                                irel, Vrel, Vup, trpnmax, kmtrpn, cmdnmax,
                                kmcmdn, F, Vj)

        carel += dt * calc_dcarel(carel, itr, irel, csqnmax, kmcsqn)

        rhs = (- calc_rhs(ina, ik1, ito, ikur, ikr, iks, ical, ipca, inak,
                            inaca, ibna, ibca))
        
        u = u + dt * rhs
        u_list[i] = u

    state_vars = typed.Dict()
    state_vars['u'] = u
    state_vars['nai'] = nai
    state_vars['ki'] = ki
    state_vars['cai'] = cai
    state_vars['caup'] = caup
    state_vars['carel'] = carel
    state_vars['m'] = m
    state_vars['h'] = h
    state_vars['j_'] = j_
    state_vars['d'] = d
    state_vars['f'] = f
    state_vars['oa'] = oa
    state_vars['oi'] = oi
    state_vars['ua'] = ua
    state_vars['ui'] = ui
    state_vars['xr'] = xr
    state_vars['xs'] = xs
    state_vars['fca'] = fca
    state_vars['irel'] = irel
    state_vars['vrel'] = vrel
    state_vars['urel'] = urel
    state_vars['wrel'] = wrel

    return u_list, state_vars
