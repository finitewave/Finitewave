import math
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
    ops = load_ops("courtemanche")
    jit_ops = wrap_calc(ops)
except KeyError as e:
    raise ImportError(
        "Courtemanche model ops not found. "
        "Install model package: pip install finitewave-model-courtemanche"
    ) from e


class CourtemancheKernel(IonicKernelGenerator):
    def __init__(self):
        super().__init__()

        self.args_order = [
            "u", "nai", "ki", "cai",
            "m", "h", "j",
            "oa", "oi", "ua", "ui", "xr", "xs", "d", "f", "fca",
            "urel", "vrel", "wrel", "irel",
            "caup", "carel",
            "nao", "ko", "cao", "R", "T", "F", "Cm",
            "gna", "gk1", "gto", "gcal", "gnab", "gcab",
            "gkr", "gks", "inakmax", "kmnai", "kmko",
            "inacamax", "kmnancx", "kmcancx", "ksatncx",
            "ipcamax", "iupmax", "kup", "caupmax",
            "krel", "Vrel", "Vup", "Vj",
            "kq10", "ibk",
            "trpnmax", "kmtrpn", "cmdnmax", "kmcmdn",
            "csqnmax", "kmcsqn",
        ]

    def generate_body(self) -> str:
        model = {var: self._indexing(var) for var in (self.arrays + self.scalars)}
        u_new = f"u_new{self._raw_indexing()}"

        return f"""\
        u_old = {model['u']}
        m_old = {model['m']}
        h_old = {model['h']}
        j_old = {model['j']}

        oa_old = {model['oa']}
        oi_old = {model['oi']}
        ua_old = {model['ua']}
        ui_old = {model['ui']}
        xr_old = {model['xr']}
        xs_old = {model['xs']}

        d_old = {model['d']}
        f_old = {model['f']}
        fca_old = {model['fca']}

        urel_old = {model['urel']}
        vrel_old = {model['vrel']}
        irel_old = {model['irel']}
        wrel_old = {model['wrel']}

        caup_old = {model['caup']}
        carel_old = {model['carel']}
        cai_old = {model['cai']}
        nai_old = {model['nai']}
        ki_old = {model['ki']}


        ena = calc_ena(nai_old, {model['nao']}, {model['R']}, {model['T']}, {model['F']})
        ek = calc_ek(ki_old, {model['ko']}, {model['R']}, {model['T']}, {model['F']})
        eca = calc_eca(cai_old, {model['cao']}, {model['R']}, {model['T']}, {model['F']})

        am = calc_am(u_old)
        bm = calc_bm(u_old)
        tau_m = calc_tau(am, bm)
        m_inf = calc_inf(am, bm)
        m_new = calc_gating_variable_rush_larsen(m_old, m_inf, tau_m, dt)

        ah = calc_ah(u_old)
        bh = calc_bh(u_old) 
        tau_h = calc_tau(ah, bh)
        h_inf = calc_inf(ah, bh)
        h_new = calc_gating_variable_rush_larsen(h_old, h_inf, tau_h, dt)

        aj = calc_aj(u_old)
        bj = calc_bj(u_old)
        tau_j = calc_tau(aj, bj)
        j_inf = calc_inf(aj, bj)
        j_new = calc_gating_variable_rush_larsen(j_old, j_inf, tau_j, dt)

        ina = calc_ina(
            u_old, m_old, h_old, j_old,
            {model['gna']}, ena, {model['Cm']}
        )

        ik1 = calc_ik1(
            u_old, {model['gk1']}, ek, {model['Cm']}
        )

        tau_oa = calc_tau_oa(u_old, {model['kq10']})
        oa_inf = calc_oa_inf(u_old)
        oa_new = calc_gating_variable_rush_larsen(oa_old, oa_inf, tau_oa, dt)

        tau_oi = calc_tau_oi(u_old, {model['kq10']})
        oi_inf = calc_oi_inf(u_old)
        oi_new = calc_gating_variable_rush_larsen(oi_old, oi_inf, tau_oi, dt)

        ito = calc_ito(
            u_old, oa_old, oi_old,
            {model['gto']}, ek, {model['Cm']}
        )

        tau_ua = calc_tau_ua(u_old, {model['kq10']})
        ua_inf = calc_ua_inf(u_old)
        ua_new = calc_gating_variable_rush_larsen(ua_old, ua_inf, tau_ua, dt)

        tau_ui = calc_tau_ui(u_old, {model['kq10']})
        ui_inf = calc_ui_inf(u_old)
        ui_new = calc_gating_variable_rush_larsen(ui_old, ui_inf, tau_ui, dt)

        ikur = calc_ikur(
            u_old, ua_old, ui_old,
            ek, {model['Cm']}
        )

        tau_xr = calc_tau_xr(u_old)
        xr_inf = calc_xr_inf(u_old)
        xr_new = calc_gating_variable_rush_larsen(xr_old, xr_inf, tau_xr, dt)

        ikr = calc_ikr(
            u_old, xr_old,
            {model['gkr']}, ek, {model['Cm']}
        )

        tau_xs = calc_tau_xs(u_old)
        xs_inf = calc_xs_inf(u_old)
        xs_new = calc_gating_variable_rush_larsen(xs_old, xs_inf, tau_xs, dt)

        iks = calc_iks(
            u_old, xs_old,
            {model['gks']}, ek, {model['Cm']}
        )

        tau_d = calc_tau_d(u_old)
        d_inf = calc_d_inf(u_old)
        d_new = calc_gating_variable_rush_larsen(d_old, d_inf, tau_d, dt)

        tau_f = calc_tau_f(u_old)
        f_inf = calc_f_inf(u_old)
        f_new = calc_gating_variable_rush_larsen(f_old, f_inf, tau_f, dt)

        tau_fca = calc_tau_fca()
        fca_inf = calc_fca_inf(cai_old)
        fca_new = calc_gating_variable_rush_larsen(fca_old, fca_inf, tau_fca, dt)

        ical = calc_ical(
            u_old, d_old, f_old,
            {model['gcal']}, fca_old, {model['Cm']}
        )

        inak = calc_inak(
            {model['inakmax']},
            nai_old, {model['nao']},
            {model['ko']}, {model['kmnai']},
            {model['kmko']}, {model['F']},
            u_old, {model['R']}, {model['T']},
            {model['Cm']}
        )

        inaca = calc_inaca(
            {model['inacamax']},
            nai_old, {model['nao']},
            cai_old, {model['cao']},
            {model['kmnancx']}, {model['kmcancx']},
            {model['ksatncx']}, {model['F']},
            u_old, {model['R']}, {model['T']},
            {model['Cm']}
        )

        ibca = calc_ibca(
            {model['gcab']}, eca, u_old, {model['Cm']}
        )

        ibna = calc_ibna(
            {model['gnab']}, ena, u_old, {model['Cm']}
        )

        ipca = calc_ipca(
            {model['ipcamax']}, cai_old, {model['Cm']}
        )

        Fn = calc_Fn(irel_old, ical, inaca, {model['F']}, {model['Vrel']})

        tau_urel = calc_tau_urel()
        urel_inf = calc_urel_inf(Fn)
        urel_new = calc_gating_variable_rush_larsen(urel_old, urel_inf, tau_urel, dt)

        tau_vrel = calc_tau_vrel(Fn)
        vrel_inf = calc_vrel_inf(Fn)
        vrel_new = calc_gating_variable_rush_larsen(vrel_old, vrel_inf, tau_vrel, dt)

        tau_wrel = calc_tau_wrel(u_old)
        wrel_inf = calc_wrel_inf(u_old)
        wrel_new = calc_gating_variable_rush_larsen(wrel_old, wrel_inf, tau_wrel, dt)

        irel_new = calc_irel(
            urel_old, vrel_old, irel_old, wrel_old,
            {model['krel']},
            carel_old, cai_old
        )

        itr = calc_itr(caup_old, carel_old)
        iup = calc_iup({model['iupmax']}, cai_old, {model['kup']})
        iupleak = calc_iupleak(caup_old, {model['caupmax']}, {model['iupmax']})

        dcaup = calc_dcaup(
            iup, iupleak, itr,
            {model['Vrel']}, {model['Vup']}
        )

        dnai = calc_dnai(
            inak, inaca, ibna, ina,
            {model['F']}, {model['Vj']}
        )

        dki = calc_dki(
            inak, ik1, ito, ikur, ikr, iks,
            {model['ibk']}, {model['F']}, {model['Vj']}
        )

        dcai = calc_dcai(
            cai_old, inaca, ipca, ical, ibca,
            iup, iupleak, irel_old,
            {model['Vrel']}, {model['Vup']},
            {model['trpnmax']}, {model['kmtrpn']},
            {model['cmdnmax']}, {model['kmcmdn']},
            {model['F']}, {model['Vj']}
        )

        dcarel = calc_dcarel(
            carel_old, itr, irel_old,
            {model['csqnmax']}, {model['kmcsqn']}
        )

        {model['m']} = m_new
        {model['h']} = h_new
        {model['j']} = j_new

        {model['oa']} = oa_new
        {model['oi']} = oi_new
        {model['ua']} = ua_new
        {model['ui']} = ui_new
        {model['xr']} = xr_new
        {model['xs']} = xs_new

        {model['d']} = d_new
        {model['f']} = f_new
        {model['fca']} = fca_new

        {model['urel']} = urel_new
        {model['vrel']} = vrel_new
        {model['irel']} = irel_new
        {model['wrel']} = wrel_new

        {model['caup']} = caup_old + dt * dcaup
        {model['nai']} = nai_old + dt * dnai
        {model['ki']} = ki_old + dt * dki
        {model['cai']} = cai_old + dt * dcai
        {model['carel']} = carel_old + dt * dcarel

        {u_new} = {u_new} + dt * -calc_rhs(
            ina, ik1, ito, ikur, ikr, iks,
            ical, ipca, inak, inaca, ibna, ibca,
            {model['Cm']}
        )
    """


class Courtemanche(CardiacModel):
    """
    This model describes the ionic currents and action potential dynamics of human atrial myocytes. 
    It includes detailed formulations for major ionic currents (fast sodium current, L-type calcium current, 
    inward rectifier potassium current, transient outward potassium current, rapid and slow delayed rectifier potassium currents, 
    and Na⁺/Ca²⁺ exchanger), as well as calcium handling mechanisms.

    The Courtemanche model is widely used as a reference atrial electrophysiology model. 
    It has served as the basis for many subsequent atrial modeling studies, including investigations of atrial fibrillation and drug effects.

    Attributes
    ----------
    D_model : float
        Diffusion coefficient for the model, used in tissue simulations.
    npfloat : str
        String specifying the floating-point precision to use (e.g., 'float64').

    Model Variables
    ---------------
    u : np.ndarray
        Transmembrane potential in mV.
    nai : np.ndarray
        Intracellular sodium concentration in mM.
    ki : np.ndarray
        Intracellular potassium concentration in mM.
    cai : np.ndarray
        Intracellular calcium concentration in mM.
    m, h, j : np.ndarray
        Gating variables for the fast sodium current.
    oa, oi : np.ndarray
        Gating variables for the transient outward potassium current (I_to).
    ua, ui : np.ndarray
        Gating variables for the ultrarapid delayed rectifier potassium current (I_kur).
    xr : np.ndarray
        Gating variable for the rapid delayed rectifier potassium current (I_kr).
    xs : np.ndarray
        Gating variable for the slow delayed rectifier potassium current (I_ks).
    d, f : np.ndarray
        Gating variables for the L-type calcium current (I_cal).
    fca : np.ndarray
        Calcium-dependent inactivation variable for I_cal.
    urel, vrel, wrel : np.ndarray
        Gating variables for sarcoplasmic reticulum (SR) calcium release.
    ire : np.ndarray
        SR calcium release current.
    caup : np.ndarray
        SR calcium uptake variable.
    carel : np.ndarray
        SR calcium release variable.

    Model Parameters
    ----------------
    nao : float
        Extracellular sodium concentration in mM.
    ko : float
        Extracellular potassium concentration in mM.
    cao : float
        Extracellular calcium concentration in mM.
    R : float
        Universal gas constant in J/(mol*K).
    T : float
        Absolute temperature in K.
    F : float
        Faraday's constant in C/mol.
    Cm : float
        Membrane capacitance in μF/cm².
    gna : float
        Maximum conductance for the fast sodium current in mS/μF.
    gk1 : float
        Maximum conductance for the inward rectifier potassium current in mS/μF.
    gto : float
        Maximum conductance for the transient outward potassium current in mS/μF.
    gcal : float
        Maximum conductance for the L-type calcium current in mS/μF.
    gnab : float
        Maximum conductance for the background sodium current in mS/μF.
    gcab : float
        Maximum conductance for the background calcium current in mS/μF.
    gkr : float
        Maximum conductance for the rapid delayed rectifier potassium current in mS/μF.
    gks : float
        Maximum conductance for the slow delayed rectifier potassium current in mS/μF.
    inakmax : float
        Maximum current for the Na⁺/K⁺ pump in μA/μF.
    kmnai : float
        Half-saturation concentration for intracellular sodium in mM (Na⁺/K⁺ pump).
    kmko : float
        Half-saturation concentration for extracellular potassium in mM (Na⁺/K⁺ pump).
    inacamax : float
        Maximum current for the Na⁺/Ca²⁺ exchanger in μA/μF.
    kmnancx : float
        Half-saturation concentration for intracellular sodium in mM (Na⁺/Ca²⁺ exchanger).
    kmcancx : float
        Half-saturation concentration for intracellular calcium in mM (Na⁺/Ca²⁺ exchanger).
    ksatncx : float
        Saturation factor for the Na⁺/Ca²⁺ exchanger.
    ipcamax : float
        Maximum current for the plasma membrane Ca²⁺ ATPase in μA/μF.
    iupmax : float
        Maximum uptake rate for the SR calcium pump in mM/ms.
    kup : float
        Half-saturation concentration for calcium uptake in mM.
    caupmax : float
        Maximum SR calcium uptake variable in mM.
    krel : float
        Rate constant for SR calcium release in ms⁻¹.
    Vrel : float
        Volume ratio for SR release.
    Vup : float
        Volume ratio for SR uptake.
    Vj : float
        Volume ratio for junctional space.
    kq10 : float
        Temperature coefficient for gating kinetics.
    ibk : float
        Background potassium current in μA/μF.
    trpnmax : float
        Maximum concentration of troponin C in mM.
    kmtrpn : float
        Half-saturation concentration for troponin C in mM.
    cmdnmax : float
        Maximum concentration of calmodulin in mM.
    kmcmdn : float
        Half-saturation concentration for calmodulin in mM.
    csqnmax : float
        Maximum concentration of calsequestrin in mM.
    kmcsqn : float
        Half-saturation concentration for calsequestrin in mM.
    

    Paper
    -----
    Courtemanche M, Ramirez RJ, Nattel S. 
    Ionic mechanisms underlying human atrial action potential properties: insights from a mathematical model. 
    Am J Physiol. 1998 Jul;275(1):H301-21.
    https://doi.org/10.1152/ajpheart.1998.275.1.H301
    """
    def __init__(self):
        super().__init__()
        self.D_model = 1.0
        self.npfloat = "float64"

        self._initialize_variables_and_parameters(ops)

    def initialize(self):
        super().initialize()

        self._allocate_state_arrays()

        gen = self._initialize_kernel(CourtemancheKernel)
    
        glb = {
            "calc_gating_variable_rush_larsen": jit_ops["calc_gating_variable_rush_larsen"],
            "calc_ena": jit_ops["calc_ena"],
            "calc_ek": jit_ops["calc_ek"],
            "calc_eca": jit_ops["calc_eca"],
            "calc_am": jit_ops["calc_am"],
            "calc_bm": jit_ops["calc_bm"],
            "calc_tau": jit_ops["calc_tau"],
            "calc_inf": jit_ops["calc_inf"],
            "calc_ah": jit_ops["calc_ah"],
            "calc_bh": jit_ops["calc_bh"],
            "calc_aj": jit_ops["calc_aj"],
            "calc_bj": jit_ops["calc_bj"],
            "calc_tau_oa": jit_ops["calc_tau_oa"],
            "calc_oa_inf": jit_ops["calc_oa_inf"],
            "calc_tau_oi": jit_ops["calc_tau_oi"],
            "calc_oi_inf": jit_ops["calc_oi_inf"],
            "calc_tau_ua": jit_ops["calc_tau_ua"],
            "calc_ua_inf": jit_ops["calc_ua_inf"],
            "calc_tau_ui": jit_ops["calc_tau_ui"],
            "calc_ui_inf": jit_ops["calc_ui_inf"],
            "calc_tau_xr": jit_ops["calc_tau_xr"],
            "calc_xr_inf": jit_ops["calc_xr_inf"],
            "calc_tau_xs": jit_ops["calc_tau_xs"],
            "calc_xs_inf": jit_ops["calc_xs_inf"],
            "calc_tau_d": jit_ops["calc_tau_d"],
            "calc_d_inf": jit_ops["calc_d_inf"],
            "calc_tau_f": jit_ops["calc_tau_f"],
            "calc_f_inf": jit_ops["calc_f_inf"],
            "calc_tau_fca": jit_ops["calc_tau_fca"],
            "calc_fca_inf": jit_ops["calc_fca_inf"],
            "calc_ina": jit_ops["calc_ina"],
            "calc_ik1": jit_ops["calc_ik1"],
            "calc_ito": jit_ops["calc_ito"],
            "calc_ikur": jit_ops["calc_ikur"],
            "calc_ikr": jit_ops["calc_ikr"],
            "calc_iks": jit_ops["calc_iks"],
            "calc_ical": jit_ops["calc_ical"],
            "calc_inak": jit_ops["calc_inak"],
            "calc_inaca": jit_ops["calc_inaca"],
            "calc_ibca": jit_ops["calc_ibca"],
            "calc_ibna": jit_ops["calc_ibna"],
            "calc_ipca": jit_ops["calc_ipca"],
            "calc_Fn": jit_ops["calc_Fn"],
            "calc_tau_urel": jit_ops["calc_tau_urel"],
            "calc_urel_inf": jit_ops["calc_urel_inf"],
            "calc_tau_vrel": jit_ops["calc_tau_vrel"],
            "calc_vrel_inf": jit_ops["calc_vrel_inf"],
            "calc_tau_wrel": jit_ops["calc_tau_wrel"],
            "calc_wrel_inf": jit_ops["calc_wrel_inf"],
            "calc_irel": jit_ops["calc_irel"],
            "calc_itr": jit_ops["calc_itr"],
            "calc_iup": jit_ops["calc_iup"],
            "calc_iupleak": jit_ops["calc_iupleak"],
            "calc_dcaup": jit_ops["calc_dcaup"],
            "calc_dnai": jit_ops["calc_dnai"],
            "calc_dki": jit_ops["calc_dki"],
            "calc_dcai": jit_ops["calc_dcai"],
            "calc_dcarel": jit_ops["calc_dcarel"],           
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

