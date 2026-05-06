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
    ops = load_ops("ten_tusscher_panfilov_2006")
    jit_ops = wrap_calc(ops)
except KeyError as e:
    raise ImportError("TP06 model ops not found.") from e


class TenTusscherPanfilov2006Kernel(IonicKernelGenerator):
    def __init__(self):
        super().__init__()
        self.args_order = [
            "u", "cai", "casr", "cass", "nai", "ki",
            "m", "h", "j", "xr1", "xr2", "xs", "r", "s",
            "d", "f", "f2", "fcass", "rr",
            "Ko", "Cao", "Nao", "Vc", "Vsr", "Vss", "Bufc", "Kbufc", 
            "Bufsr", "Kbufsr", "Bufss", "Kbufss", "Vmaxup", "Kup", "Vrel", "k1", "k2", "k3", "k4",
            "EC", "maxsr", "minsr", "Vleak", "Vxfer", "R", "F", "T", "Cm",
            "gKr", "gKs", "gK1", "gto", "gNa", "gbNa", "gCaL", "gbCa", "gpCa", "KpCa", "gpK", "pKNa",
            "KmK", "KmNa", "pNaK", "kNaCa", "KmNai", "KmCa", "ksat", "gamma", "alpha"
        ]

    def generate_body(self) -> str:
        model = {name: self._indexing(name) for name in self.args_order}
        u_new = f"u_new{self._raw_indexing()}"

        return f"""\
        # inverseVcF2 = 1.0 / (2 * {model['Vc']} * {model['F']})
        # inverseVcF = 1.0 / ({model['Vc']} * {model['F']})
        # inversevssF2 = 1.0 / (2 * {model['Vss']} * {model['F']})

        # Old state
        u_old = {model['u']}
        m_old = {model['m']}
        h_old = {model['h']}
        j_old = {model['j']}
        d_old = {model['d']}
        f_old = {model['f']}
        f2_old = {model['f2']}
        fcass_old = {model['fcass']}
        r_old = {model['r']}
        s_old = {model['s']}
        xr1_old = {model['xr1']}
        xr2_old = {model['xr2']}
        xs_old = {model['xs']}
        rr_old = {model['rr']}

        casr_old = {model['casr']}
        cass_old = {model['cass']}
        cai_old = {model['cai']}
        nai_old = {model['nai']}
        ki_old = {model['ki']}

        Ek = calc_Ek({model['Ko']}, ki_old, {model['R']}, {model['T']}, {model['F']})
        Ena = calc_Ena({model['Nao']}, nai_old, {model['R']}, {model['T']}, {model['F']})
        Eks = calc_Eks({model['Ko']}, ki_old, {model['Nao']}, nai_old, {model['pKNa']}, {model['R']}, {model['T']}, {model['F']})
        Eca = calc_Eca({model['Cao']}, cai_old, {model['R']}, {model['T']}, {model['F']})

        m_inf = calc_m_inf(u_old)
        tau_m = calc_tau_m(u_old)
        m_new = calc_gating_variable_rush_larsen(m_old, m_inf, tau_m, dt)

        h_inf = calc_h_inf(u_old)
        tau_h = calc_tau_h(u_old)
        h_new = calc_gating_variable_rush_larsen(h_old, h_inf, tau_h, dt)

        j_inf = h_inf
        tau_j = calc_tau_j(u_old)
        j_new = calc_gating_variable_rush_larsen(j_old, j_inf, tau_j, dt)
        
        ina = calc_ina(u_old, m_old, h_old, j_old, {model['gNa']}, Ena)

        d_inf = calc_d_inf(u_old)
        tau_d = calc_tau_d(u_old)
        d_new = calc_gating_variable_rush_larsen(d_old, d_inf, tau_d, dt)

        f_inf = calc_f_inf(u_old)
        tau_f = calc_tau_f(u_old)
        f_new = calc_gating_variable_rush_larsen(f_old, f_inf, tau_f, dt)

        f2_inf = calc_f2_inf(u_old)
        tau_f2 = calc_tau_f2(u_old)
        f2_new = calc_gating_variable_rush_larsen(f2_old, f2_inf, tau_f2, dt)

        fcass_inf = calc_fcass_inf(cass_old)
        tau_fcass = calc_tau_fcass(cass_old)
        fcass_new = calc_gating_variable_rush_larsen(fcass_old, fcass_inf, tau_fcass, dt)

        ical = calc_ical(
            u_old, d_old, f_old, f2_old,
            fcass_old, {model['Cao']}, cass_old,
            {model['gCaL']}, {model['F']},
            {model['R']}, {model['T']}
        )

        r_inf = calc_r_inf(u_old)
        tau_r = calc_tau_r(u_old)
        r_new = calc_gating_variable_rush_larsen(r_old, r_inf, tau_r, dt)

        s_inf = calc_s_inf(u_old)
        tau_s = calc_tau_s(u_old)
        s_new = calc_gating_variable_rush_larsen(s_old, s_inf, tau_s, dt)

        ito = calc_ito(u_old, r_old, s_old, Ek, {model['gto']})

        xr1_inf = calc_xr1_inf(u_old)
        tau_xr1 = calc_tau_xr1(u_old)
        xr1_new = calc_gating_variable_rush_larsen(xr1_old, xr1_inf, tau_xr1, dt)

        xr2_inf = calc_xr2_inf(u_old)
        tau_xr2 = calc_tau_xr2(u_old)
        xr2_new = calc_gating_variable_rush_larsen(xr2_old, xr2_inf, tau_xr2, dt)

        ikr = calc_ikr(
            u_old, xr1_old, xr2_old, Ek,
            {model['gKr']}, {model['Ko']}
        )

        xs_inf = calc_xs_inf(u_old)
        tau_xs = calc_tau_xs(u_old)
        xs_new = calc_gating_variable_rush_larsen(xs_old, xs_inf, tau_xs, dt)

        iks = calc_iks(u_old, xs_old, Eks, {model['gKs']})

        ik1 = calc_ik1(u_old, Ek, {model['gK1']})

        inaca = calc_inaca(
            u_old, {model['Nao']}, nai_old,
            {model['Cao']}, cai_old,
            {model['KmNai']}, {model['KmCa']},
            {model['kNaCa']}, {model['ksat']},
            {model['gamma']},
            {model['alpha']}, {model['F']},
            {model['R']}, {model['T']}
        )

        inak = calc_inak(
            u_old, nai_old, {model['Ko']},
            {model['KmK']}, {model['KmNa']},
            {model['pNaK']}, {model['F']},
            {model['R']}, {model['T']}
        )

        ipca = calc_ipca(
            cai_old, {model['KpCa']},
            {model['gpCa']}
        )

        ipk = calc_ipk(
            u_old, Ek,
            {model['gpK']}
        )

        ibna = calc_ibna(
            u_old, Ena,
            {model['gbNa']}
        )

        ibca = calc_ibca(
            u_old, Eca,
            {model['gbCa']}
        )

        kCaSR = calc_kCaSR(
            casr_old, {model['maxsr']}, 
            {model['minsr']}, {model['EC']}
        )
        k1_ = {model['k1']}/kCaSR
        k2_ = {model['k2']}*kCaSR
        drr = {model['k4']}*(1-rr_old)-k2_*cass_old*rr_old
        rr_new = rr_old + dt*drr
    
        oo_old = k1_*cass_old*cass_old * rr_old/({model['k3']}+k1_*cass_old*cass_old)

        irel = calc_irel(
            oo_old, casr_old, cass_old,
            {model['Vrel']})

        ileak = calc_ileak(
            casr_old, cai_old,
            {model['Vleak']}
        )

        iup = calc_iup(
            cai_old,
            {model['Vmaxup']}, {model['Kup']}
        )

        ixfer = calc_ixfer(
            cass_old, cai_old,
            {model['Vxfer']}
        )

        # Concentration updates from old state and old-state currents
        casr_new = calc_casr(
            dt, casr_old,
            {model['Bufsr']}, {model['Kbufsr']},
            iup, irel, ileak
        )

        cass_new = calc_cass(
            dt, cass_old,
            {model['Bufss']}, {model['Kbufss']},
            ixfer, irel, ical,
            {model['Cm']},
            {model['Vc']}, {model['Vss']},
            {model['Vsr']}, {model['F']}
        )

        cai_new = calc_cai(
            dt, cai_old,
            {model['Bufc']}, {model['Kbufc']},
            ibca, ipca, inaca, iup, ileak, ixfer,
            {model['Cm']},
            {model['Vsr']}, {model['Vc']},
            {model['F']}
        )

        dnai = calc_dnai(
            ina, ibna,
            inak, inaca,
            {model['Cm']},
            {model['Vc']}, {model['F']}
        )

        dki = calc_dki(
            ik1, ito,
            ikr, iks,
            inak, ipk,
            {model['Cm']},
            {model['Vc']}, {model['F']}
        )

        # Commit new state
        {model['m']} = m_new
        {model['h']} = h_new
        {model['j']} = j_new

        {model['d']} = d_new
        {model['f']} = f_new
        {model['f2']} = f2_new
        {model['fcass']} = fcass_new

        {model['r']} = r_new
        {model['s']} = s_new
        {model['xr1']} = xr1_new
        {model['xr2']} = xr2_new
        {model['xs']} = xs_new

        {model['rr']} = rr_new

        {model['casr']} = casr_new
        {model['cass']} = cass_new
        {model['cai']} = cai_new
        {model['nai']} = nai_old + dt * dnai
        {model['ki']} = ki_old + dt * dki

        {u_new} = {u_new} + dt * -calc_rhs(
            ikr, iks,
            ik1, ito,
            ina, ibna,
            ical, ibca,
            inak, inaca,
            ipca, ipk
        )
"""


class TenTusscherPanfilov2006(CardiacModel):
    """
    Implements the ten Tusscher–Panfilov 2006 (TP06) human ventricular ionic model in 2D.

    The TP06 model is a detailed biophysical model of the human ventricular 
    action potential, designed to simulate realistic electrical behavior in 
    tissue including alternans, reentrant waves, and spiral wave breakup.

    This model includes:
    - Dynamic state variables (voltage, ion concentrations, channel gates, buffers)
    - Full calcium handling with subspace (cass) and sarcoplasmic reticulum (casr)
    - Sodium, potassium, and calcium currents including background, exchanger, and pumps
    - Buffering effects and intracellular transport

    Attributes
    ----------
    D_model : float
        Diffusion coefficient specific to this model (cm²/ms).
    npfloat : str
        String specifying the floating-point precision to use (e.g., 'float64').

    Model Variables
    ---------------
    u : np.ndarray
        Transmembrane potential (mV).
    cai : np.ndarray
        Intracellular calcium concentration (mM).
    casr : np.ndarray
        Calcium concentration in the sarcoplasmic reticulum (mM).
    cass : np.ndarray
        Calcium concentration in the subsarcolemmal space (mM).
    nai : np.ndarray
        Intracellular sodium concentration (mM).
    ki : np.ndarray
        Intracellular potassium concentration (mM).
    m, h, j : np.ndarray
        Gating variables for the fast sodium current.
    xr1, xr2 : np.ndarray
        Gating variables for the rapid delayed rectifier K⁺ current.
    xs : np.ndarray
        Gating variable for the slow delayed rectifier K⁺ current.
    r, s : np.ndarray
        Gating variables for the transient outward K⁺ current.
    d, f, f2, fcass : np.ndarray
        Gating variables for the L-type calcium current.
    rr, oo : np.ndarray
        Gating variables for the ryanodine receptor (calcium release channel).
    
    Model Parameters
    ----------------
    ko : float
        Extracellular potassium concentration (mM).
    cao : float
        Extracellular calcium concentration (mM).
    nao : float
        Extracellular sodium concentration (mM).
    Vc : float
        Cytoplasmic volume (μL).
    Vsr : float
        Sarcoplasmic reticulum volume (μL).
    Vss : float
        Subsarcolemmal space volume (μL).
    R, T, F : float
        Universal gas constant, absolute temperature, and Faraday constant.
    RTONF : float
        Precomputed RT/F value for Nernst equation.
    Cm : float
        Membrane capacitance per unit area (μF/cm²).
    gna, gcal, gkr, gks, gk1, gto : float
        Conductances for major ionic channels.
    gbna, gbca : float
        Background sodium and calcium conductances.
    gpca, gpk : float
        Pump-related conductances.
    knak, knaca : float
        Maximal Na⁺/K⁺ pump and Na⁺/Ca²⁺ exchanger rates.
    Km*, Kbuf*, Vmaxup, Vrel, etc.
        Numerous kinetic constants for buffering, pump activity, and calcium handling.

    Paper
    -----
    ten Tusscher KH, Panfilov AV. 
    Alternans and spiral breakup in a human ventricular tissue model.
    Am J Physiol Heart Circ Physiol. 2006 Sep;291(3):H1088–H1100.
    https://doi.org/10.1152/ajpheart.00109.2006

    """
    def __init__(self):
        super().__init__()

        self.D_model = 0.154
        self.npfloat = "float64"

        self._initialize_variables_and_parameters(ops)

    def initialize(self):
        super().initialize()

        self._allocate_state_arrays()

        gen = self._initialize_kernel(TenTusscherPanfilov2006Kernel)

        glb = {
            "calc_gating_variable_rush_larsen": jit_ops["calc_gating_variable_rush_larsen"],
            "calc_Ek": jit_ops["calc_Ek"],
            "calc_Ena": jit_ops["calc_Ena"],
            "calc_Eks": jit_ops["calc_Eks"],
            "calc_Eca": jit_ops["calc_Eca"],
            "calc_m_inf": jit_ops["calc_m_inf"],
            "calc_tau_m": jit_ops["calc_tau_m"],
            "calc_h_inf": jit_ops["calc_h_inf"],
            "calc_tau_h": jit_ops["calc_tau_h"],
            "calc_tau_j": jit_ops["calc_tau_j"],
            "calc_d_inf": jit_ops["calc_d_inf"],
            "calc_tau_d": jit_ops["calc_tau_d"],
            "calc_f_inf": jit_ops["calc_f_inf"],
            "calc_tau_f": jit_ops["calc_tau_f"],
            "calc_f2_inf": jit_ops["calc_f2_inf"],
            "calc_tau_f2": jit_ops["calc_tau_f2"],
            "calc_fcass_inf": jit_ops["calc_fcass_inf"],
            "calc_tau_fcass": jit_ops["calc_tau_fcass"],
            "calc_ical": jit_ops["calc_ical"],
            "calc_r_inf": jit_ops["calc_r_inf"],
            "calc_tau_r": jit_ops["calc_tau_r"],
            "calc_s_inf": jit_ops["calc_s_inf"],
            "calc_tau_s": jit_ops["calc_tau_s"],
            "calc_xr1_inf": jit_ops["calc_xr1_inf"],
            "calc_tau_xr1": jit_ops["calc_tau_xr1"],
            "calc_xr2_inf": jit_ops["calc_xr2_inf"],
            "calc_tau_xr2": jit_ops["calc_tau_xr2"],
            "calc_xs_inf": jit_ops["calc_xs_inf"],
            "calc_tau_xs": jit_ops["calc_tau_xs"],
            "calc_ina": jit_ops["calc_ina"],
            "calc_ito": jit_ops["calc_ito"],
            "calc_ikr": jit_ops["calc_ikr"],
            "calc_iks": jit_ops["calc_iks"],
            "calc_ik1": jit_ops["calc_ik1"],
            "calc_inaca": jit_ops["calc_inaca"],
            "calc_inak": jit_ops["calc_inak"],
            "calc_ipca": jit_ops["calc_ipca"],
            "calc_ipk": jit_ops["calc_ipk"],
            "calc_ibna": jit_ops["calc_ibna"],
            "calc_ibca": jit_ops["calc_ibca"],
            "calc_kCaSR": jit_ops["calc_kCaSR"],
            "calc_irel": jit_ops["calc_irel"],
            "calc_ileak": jit_ops["calc_ileak"],
            "calc_iup": jit_ops["calc_iup"],
            "calc_ixfer": jit_ops["calc_ixfer"],
            "calc_casr": jit_ops["calc_casr"],
            "calc_cass": jit_ops["calc_cass"],
            "calc_cai": jit_ops["calc_cai"],
            "calc_dnai": jit_ops["calc_dnai"],
            "calc_dki": jit_ops["calc_dki"],
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
        if cardiac_tissue.fibers is None:
            if cardiac_tissue.dimensions == 2:
                return IsotropicStencil2D()
            if cardiac_tissue.dimensions == 3:
                return IsotropicStencil3D()
            raise ValueError("Unsupported number of dimensions")
        else:
            if cardiac_tissue.dimensions == 2:
                return AsymmetricStencil2D()
            if cardiac_tissue.dimensions == 3:
                return AsymmetricStencil3D()
            raise ValueError("Unsupported number of dimensions")
