import numpy as np
from numba import njit, prange

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import (
    AsymmetricStencil2D
)
from finitewave.cpuwave2D.stencil.isotropic_stencil_2d import (
    IsotropicStencil2D
)


class Courtemanche2D(CardiacModel):
    """
    A class to represent the Courtemanche cardiac model in 2D.

    Attributes
    ----------
    D_model : float
        Model specific diffusion coefficient.
    state_vars : list of str
        List of state variable names.
    """

    def __init__(self):
        super().__init__()
        self.D_model = 0.154
        self.state_vars = ["u", "nai", "ki", "cai", "caup", "carel", "m", "h", "j_",
         "d", "f", "oa", "oi", "ua", "ui", "xr", "xs", "fca", "irel", "vrel", "urel", "wrel"]
        
        self.npfloat = 'float64'

    def initialize(self):
        """
        Initializes the model's state variables and diffusion/ionic kernels.

        Sets up the initial values for membrane potential, ion concentrations,
        gating variables, and assigns the appropriate kernel functions.
        """
        super().initialize()
        shape = self.cardiac_tissue.mesh.shape

        self.u = -84.5*np.ones(shape, dtype=self.npfloat)      # (mV)
        self.u_new = self.u.copy()                             # (mV)
        self.nai = 11.2*np.ones(shape, dtype=self.npfloat)     # (mM)
        self.ki = 139*np.ones(shape, dtype=self.npfloat)       # (mM)
        self.cai = 0.000102*np.ones(shape, dtype=self.npfloat) # (mM)
        self.caup = 1.6*np.ones(shape, dtype=self.npfloat)     # (mM)
        self.carel = 1.1*np.ones(shape, dtype=self.npfloat)    # (mM)
        self.m = 0.00291*np.ones(shape, dtype=self.npfloat)    
        self.h = 0.965*np.ones(shape, dtype=self.npfloat)      
        self.j_ = 0.978*np.ones(shape, dtype=self.npfloat)
        self.d = 0.000137*np.ones(shape, dtype=self.npfloat)
        self.f = 0.999837*np.ones(shape, dtype=self.npfloat)
        self.oa = 0.000592*np.ones(shape, dtype=self.npfloat)
        self.oi = 0.9992*np.ones(shape, dtype=self.npfloat)
        self.ua = 0.003519*np.ones(shape, dtype=self.npfloat)
        self.ui = 0.9987*np.ones(shape, dtype=self.npfloat)
        self.xs = 0.0187*np.ones(shape, dtype=self.npfloat)
        self.xr = 0.0000329*np.ones(shape, dtype=self.npfloat)
        self.fca = 0.775*np.ones(shape, dtype=self.npfloat)
        self.irel = 0*np.ones(shape, dtype=self.npfloat)
        self.vrel = 1*np.ones(shape, dtype=self.npfloat)
        self.urel = 0*np.ones(shape, dtype=self.npfloat)
        self.wrel = 0.9*np.ones(shape, dtype=self.npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel function to update ionic currents and state
        variables
        """
        ionic_kernel_2d(self.u_new, self.u, self.nai, self.ki, self.cai, self.caup, self.carel, 
                        self.m, self.h, self.j_, self.d, self.f, self.oa, self.oi, self.ua, 
                        self.ui, self.xr, self.xs, self.fca, self.irel, self.vrel, self.urel, 
                        self.wrel, self.cardiac_tissue.myo_indexes, self.dt)

    def select_stencil(self, cardiac_tissue):
        """
        Selects the appropriate stencil for diffusion based on the tissue
        properties. If the tissue has fiber directions, an asymmetric stencil
        is used; otherwise, an isotropic stencil is used.

        Parameters
        ----------
        cardiac_tissue : CardiacTissue2D
            A tissue object representing the cardiac tissue.

        Returns
        -------
        Stencil
            The stencil object to use for diffusion computations.
        """
        if cardiac_tissue.fibers is None:
            return IsotropicStencil2D()

        return AsymmetricStencil2D()

@njit
def safe_exp(x):
    """
    Clamps the value of x between -500 and 500 and computes exp(x).
    """
    # if x > 500:
    #     x = 500
    # elif x < -500:
    #     x = -500
    return np.exp(x)

@njit
def calc_cmdn(cmdnmax, kmcmdn, cai):
    """
    Calculates the concentration of calmodulin.
    """
    cmdn = cmdnmax*cai/(cai + kmcmdn)
    return cmdn

@njit
def calc_trpn(trpnmax, kmtrpn, cai):
    """
    Calculates the concentration of troponin.
    """
    trpn = trpnmax*cai/(cai + kmtrpn)
    return trpn

@njit
def calc_csqn(csqnmax, kmcsqn, carel):
    """
    Calculates the concentration of calsequestrin.
    """
    csqn = csqnmax*carel/(carel + kmcsqn)
    return csqn

@njit
def calc_nai(nai, dt, inak, inaca, ibna, ina, F, Vj):
    """
    Calculates the intracellular sodium concentration.
    """
    dnai = (-3*inak-3*inaca - ibna - ina)/(F*Vj)
    nai += dt*dnai
    return nai

@njit
def calc_ki(ki, dt, inak, ik1, ito, ikur, ikr, iks, ibk, F, Vj):
    """
    Calculates the intracellular potassium concentration.
    """
    dki = (2*inak - ik1 - ito - ikur - ikr - iks - ibk)/(F*Vj)
    ki += dt*dki
    return ki

@njit
def calc_cai(cai, dt, inaca, ipca, ical, ibca, iup, iupleak, irel, Vrel, vup, trpnmax, kmtrpn, cmdnmax, kmcmdn, F, Vj): 
    """
    Calculates the intracellular calcium concentration.
    """
    B1 = (2*inaca - ipca - ical - ibca)/(2*F*Vj) + (vup*(iupleak - iup) + irel*Vrel)/Vj
    B2 = 1 + (trpnmax*kmtrpn)/((cai + kmtrpn)**2) + (cmdnmax*kmcmdn)/((cai + kmcmdn)**2)
    dcai = B1/B2
    cai += dt*dcai
    # print ("CAI:", cai)
    return cai

@njit
def calc_caup(caup, dt, iup, iupleak, itr, Vrel, vup):
    """
    Calculates the calcium concentration in the up compartment.
    """
    dcaup = iup - iupleak - itr*(Vrel/vup)
    caup += dt*dcaup
    return caup

@njit
def calc_carel(carel, dt, itr, irel, csqnmax, kmcsqn):
    """
    Calculates the calcium concentration in the release compartment.
    """
    dcarel = (itr - irel)/(1 + (csqnmax*kmcsqn)/((carel + kmcsqn)**2))
    carel += dt*dcarel
    return carel

@njit
def calc_equilibrum_potentials(nai, nao, ki, ko, cai, cao, R, T, F):
    """
    Calculates the equilibrum potentials for the cell.
    """
    # nai = np.maximum(nai, 1e-8)
    # nao = np.maximum(nao, 1e-8)
    ena = (R*T/F)*np.log(nao/nai)

    # ki = np.maximum(ki, 1e-8)
    # ko = np.maximum(ko, 1e-8)
    ek = (R*T/F)*np.log(ko/ki)
    
    # cai = np.maximum(cai, 1e-8)
    # cao = np.maximum(cao, 1e-8)
    eca = (R*T/(2*F))*np.log(cao/cai)
    return ena, ek, eca

@njit
def calc_ina(u, m, h, j, gna, ena):
    """
    Calculates the fast sodium current.
    """
    ina = gna*(m**3)*h*j*(u - ena)
    return ina

@njit
def calc_gating_m(m, u, dt):
    """
    Calculates the gating variable m for the fast sodium current.
    """
    am = 0
    if u == -47.13:
        am = 3.2
    else:
        am = 0.32 * (u + 47.13) / (1 - safe_exp(-0.1 * (u + 47.13)))

    bm = 0.08*safe_exp(-u/11)
    m_inf = am/(am + bm)
    tau_m = 1/(am + bm)
    m = m_inf - (m_inf - m)*safe_exp(-dt/tau_m)
    return m

@njit
def calc_gating_h(h, u, dt):
    """
    Calculates the gating variable h for the fast sodium current.
    """
    ah = 0.135*safe_exp(-(80 + u)/6.8)
    bh = 3.56*safe_exp(0.079*u) + 310000*safe_exp(0.35*u)
    if u >= -40:
        ah = 0    
        bh = 1/(0.13*(1 + safe_exp(-(u + 10.66)/11.1)))
    
    h_inf = ah/(ah + bh)
    tau_h = 1/(ah + bh)
    h = h_inf - (h_inf - h)*safe_exp(-dt/tau_h)
    return h

@njit
def calc_gating_j(j, u, dt):
    """
    Calculates the gating variable j for the fast sodium current.
    """
    
    aj = (-127140*safe_exp(0.2444*u) - 0.00003474*safe_exp(-0.04391*u))*(u + 37.78)/(1 + safe_exp(0.311*(u + 79.23)))
    bj = 0.1212*safe_exp(-0.01052*u)/(1 + safe_exp(-0.1378*(u + 40.14)))
    if u >= -40:
        aj = 0
        bj = 0.3*safe_exp(-0.0000002535*u)/(1 + safe_exp(-0.1*(u + 32)))
    j_inf = aj/(aj + bj)
    tau_j = 1/(aj + bj)
    j = j_inf - (j_inf - j)*safe_exp(-dt/tau_j)
    return j

@njit
def calc_ik1(u, gk1, ek):
    """
    Calculates the time-independent potassium current.
    """
    ik1 = gk1*(u - ek)/(1 + safe_exp(0.07*(u + 80)))
    return ik1

@njit
def calc_ito(u, dt, kq10, oa, oi, gto, ek):
    """
    Calculates the transient outward potassium current.
    """
    ao = 0.65/(safe_exp(-(u + 10)/8.5) + safe_exp(-(u - 30)/59.0))
    bo = 0.65/(2.5 + safe_exp((u + 82)/17.0))

    tau_o = 1/(kq10*(ao + bo))
    o_inf = 1/(1 + safe_exp(-(u + 20.47)/17.54))

    aoi = 1/(18.53 + safe_exp((u + 113.7)/10.95))
    boi = 1/(35.56 + safe_exp(-(u + 1.26)/7.44))

    tau_oi = 1/(kq10*(aoi + boi))
    oi_inf = 1/(1 + safe_exp((u + 43.1)/5.3))

    oa = o_inf - (o_inf - oa)*safe_exp(-dt/tau_o)
    oi = oi_inf - (oi_inf - oi)*safe_exp(-dt/tau_oi)

    ito = gto*(oa**3)*oi*(u - ek)  

    return ito, oa, oi

@njit
def calc_ikur(u, dt, kq10, ua, ui, ek):
    """
    Calculates the ultra-rapid delayed rectifier potassium current.
    """
    gkur = 0.005 + 0.05/(1 + safe_exp(-(u - 15)/13.0))

    # print ("Calcs inside ikur step by step")

    # print ("gkur: ", gkur)

    aua = 0.65/(safe_exp(-(u + 10)/8.5) + safe_exp(-(u - 30)/59.0))
    # print ("aua: ", aua)
    bua = 0.65/(2.5 + safe_exp((u + 82)/17.0))
    # print ("bua: ", bua)
    tau_ua = 1/(kq10*(aua + bua))
    # print ("tau_u: ", tau_u)
    ua_inf = 1/(1 + safe_exp(-(u + 30.3)/9.6))
    # print ("ua_inf: ", ua_inf)
    aui = 1/(21 + safe_exp(-(u - 185)/28.0))
    # print ("aui: ", aui)
    bui = safe_exp((u - 158)/16.0)
    # print ("bui: ", bui)

    tau_ui = 1/(kq10*(aui + bui))
    # print ("tau_ui: ", tau_ui)
    ui_inf = 1/(1 + safe_exp((u - 99.45)/27.48))
    # print ("ui_inf: ", ui_inf)

    ua = ua_inf - (ua_inf - ua)*safe_exp(-dt/tau_ua)
    # print ("ua: ", ua)
    ui = ui_inf - (ui_inf - ui)*safe_exp(-dt/tau_ui)
    # print ("ui: ", ui)

    ikur = gkur*(ua**3)*ui*(u - ek)
    # print ("ikur: ", ikur)

    return ikur, ua, ui

@njit
def calc_ikr(u, dt, xr, ek):
    """
    Calculates the rapid delayed rectifier potassium current.
    """
    gkr = 0.0294 # * np.sqrt(ko / 5.4)
    axr = 0.0003*(u + 14.1)/(1 - safe_exp(-(u + 14.1)/5))
    bxr = 0.000073898*(u - 3.3328)/(safe_exp((u - 3.3328)/5.1237) - 1)

    tau_xr = 1/(axr + bxr)
    xr_inf = 1/(1 + safe_exp(-(u + 14.1)/6.5))

    xr = xr_inf - (xr_inf - xr)*safe_exp(-dt/tau_xr)

    ikr = (gkr*xr*(u - ek))/(1 + safe_exp((u + 15)/22.4))

    return ikr, xr

@njit
def calc_iks(u, dt, xs, gks, ek):
    """
    Calculates the slow delayed rectifier potassium current.
    """
    axs = 0.00004*(u - 19.9)/(1 - safe_exp(-(u - 19.9)/17))
    bxs = 0.000035*(u - 19.9)/(safe_exp((u - 19.9)/9) - 1)

    tau_xs = 1/(2*(axs + bxs))
    xs_inf = 1/np.sqrt(1 + safe_exp(-(u - 19.9)/12.7))

    xs = xs_inf - (xs_inf - xs)*safe_exp(-dt/tau_xs)

    iks = gks*(xs**2)*(u - ek)

    return iks, xs

@njit
def calc_ical(u, dt, d, f, cai, gcal, fca):
    """
    Calculates the L-type calcium current.
    """
    tau_d = (1 - safe_exp(-(u + 10)/6.24))/(0.035*(u + 10)*(1 + safe_exp(-(u + 10)/6.24)))
    d_inf = 1/(1 + safe_exp(-(u + 10)/8.0))

    tau_f = 9/(0.0197*safe_exp(-(0.0337**2)*((u + 10)**2)) + 0.02)
    f_inf = 1/(1 + safe_exp((u + 28)/6.9))

    tau_fca = 2
    fca_inf = 1/(1 + cai/0.00035)

    d = d_inf - (d_inf - d)*safe_exp(-dt/tau_d)
    f = f_inf - (f_inf - f)*safe_exp(-dt/tau_f)
    fca = fca_inf - (fca_inf - fca)*safe_exp(-dt/tau_fca)

    ical = gcal*d*f*fca*(u - 65)

    return ical, d, f, fca

@njit
def calc_inak(inakmax, nai, nao, ko, kmnai, kmko, F, u, R, T):
    """
    Calculates the sodium-potassium pump current.
    """
    s = (1/7.0)*(safe_exp(nao/67.3) - 1)
    fnak = 1/(1 + 0.1245*safe_exp(-0.1*(F*u)/(R*T)) + 0.0365*s*safe_exp(-(F*u)/(R*T)))

    inak = inakmax*fnak*(1/(1 + (kmnai/nai)**1.5))*(ko/(ko + kmko))

    return inak

@njit
def calc_inaca(inacamax, nai, nao, cai, cao, kmnancx, kmcancx, ksatncx, F, u, R, T):
    """
    Calculates the sodium-calcium exchanger current.
    """
    gamma = 0.35

    # Clamp concentrations to avoid numerical issues
    # nai = max(nai, 1e-8)
    # nao = max(nao, 1e-8)
    # cai = max(cai, 1e-8)
    # cao = max(cao, 1e-8)

    # Exponential terms with clamping
    exp_term = safe_exp(gamma * (F * u) / (R * T))
    exp_rev_term = safe_exp((gamma - 1) * (F * u) / (R * T))

    # Numerator
    numerator = inacamax * (exp_term * nai**3 * cao - exp_rev_term * nao**3 * cai)

    # Denominator
    term1 = (kmnancx**3 + nao**3)  # (K_m,Na^3 + [Na+]_i^3)
    term2 = (kmcancx + cao)        # (K_m,Ca + [Ca2+]_o)
    term3 = (1 + ksatncx * exp_rev_term)  # (1 + k_sat * exp(...))

    denominator = term1 * term2 * term3

    # Avoid division by zero
    # denominator = max(denominator, 1e-8)

    # Calculate INaCa
    inaca = numerator / denominator

    # Clamp INaCa to physiological limits
    # inaca = max(min(inaca, 1000), -1000)

    # Debugging information
    # print("INaCa Debug:", inaca, "Numerator:", numerator, "Denominator:", denominator)

    return inaca


@njit
def calc_ibca(gcab, eca, u):
    """
    Calculates the background calcium current.
    """
    ibca = gcab*(u - eca)
    return ibca

@njit
def calc_ibna(gnab, ena, u):
    """
    Calculates the background sodium current.
    """
    ibna = gnab*(u - ena)
    return ibna

@njit
def calc_ipca(ipcamax, cai):
    """
    Calculates the sarcolemmal calcium pump current.
    """
    ipca = ipcamax*cai/(cai + 0.0005)
    return ipca

@njit
def calc_irel(dt, urel, vrel, irel, wrel, ical, inaca, krel, carel, cai, u, F, Vrel): 
    """
    Calculates the calcium release from the JSR.
    """
    tau_u = 8

    Fn = 1e-12*Vrel*irel - ((5*1e-13)/F)*(0.5*ical - 0.2*inaca) 

    u_inf = 1/(1 + safe_exp(-(Fn - 3.4175e-13)/13.67e-16))

    tau_v = 1.91 + 2.09/(1 + safe_exp(-(Fn - 3.4175e-13)/13.67e-16))
    v_inf = 1 - 1/(1 + safe_exp(-(Fn - 6.835e-14)/13.67e-16))

    tau_w = 6 * (1 - safe_exp(-(u - 7.9) / 5.0)) / ((1 + 0.3 * safe_exp(-(u - 7.9) / 5.0)) * (u - 7.9))
    w_inf = 1 - 1/(1 + safe_exp(-(u - 40)/17.0))

    urel = u_inf - (u_inf - urel)*safe_exp(-dt/tau_u)
    vrel = v_inf - (v_inf - vrel)*safe_exp(-dt/tau_v)
    wrel = w_inf - (w_inf - wrel)*safe_exp(-dt/tau_w)

    irel = krel*(urel**2)*vrel*wrel*(carel - cai)

    return irel, urel, vrel, wrel

@njit
def calc_itr(caup, carel):
    """
    Calculates the transfer of calcium from the NSR to the JSR.
    """
    tautr = 180
    itr = (caup - carel)/tautr
    return itr

@njit
def calc_iup(iupmax, cai, kup):
    """
    Calculates the uptake of calcium into the NSR.
    """
    iup = iupmax/(1 + (kup/cai))
    return iup

@njit
def calc_iupleak(caup, caupmax, iupmax):
    """
    Calculates the leak of calcium from the NSR.
    """
    iupleak = (caup/caupmax)*iupmax
    return iupleak



@njit(parallel=True)
def ionic_kernel_2d(u_new, u, nai, ki, cai, caup, carel, m, h, j_, d, f, oa, oi, ua, ui, xs, xr, fca, irel, vrel, urel, wrel, indexes, dt):
    n_i = u.shape[0]
    n_j = u.shape[1]

    gna = 7.8
    gnab = 0.000674

    gk1 = 0.09
    gks = 0.129
    gto = 0.1652
    gcal = 0.1238
    gcab = 0.00113

    F = 96485.0
    T = 310.0
    R = 8314.0 

    Vc = 20100
    Vj = Vc*0.68     # (uL)
    vup = Vj*0.06*0.92
    Vrel = Vj*0.06*0.08

    ibk = 0.0
    cao = 1.8 # mM
    nao = 140 # mM
    ko  = 5.4 # mM

    caupmax =  15 # mM/ms
    kup = 0.00092 # mM

    kmnai = 10
    kmko = 1.5
    kmnancx = 87.5
    kmcancx = 1.38
    ksatncx = 0.1

    kmcmdn = 0.00238
    kmtrpn = 0.0005
    kmcsqn = 0.8

    trpnmax = 0.07 # mM
    cmdnmax = 0.05 # mM
    csqnmax = 10.0 # mM
    inacamax = 1600
    inakmax = 0.6
    ipcamax = 0.275
    krel = 30

    iupmax = 0.005

    kq10 = 3

    # print(f"nai: {nai[5, 5]}")
    # print(f"ki: {ki[5, 5]}")
    # print(f"cai: {cai[5, 5]}")
    # print(f"caup: {caup[5, 5]}")
    # print(f"carel: {carel[5, 5]}")
    # print(f"m: {m[5, 5]}")
    # print(f"h: {h[5, 5]}")
    # print(f"j: {j_[5, 5]}")
    # print(f"d: {d[5, 5]}")
    # print(f"f: {f[5, 5]}")
    # print(f"oa: {oa[5, 5]}")
    # print(f"oi: {oi[5, 5]}")
    # print(f"ua: {ua[5, 5]}")
    # print(f"ui: {ui[5, 5]}")
    # print(f"xs: {xs[5, 5]}")
    # print(f"xr: {xr[5, 5]}")
    # print(f"fca: {fca[5, 5]}")
    # print(f"irel: {irel[5, 5]}")
    # print(f"vrel: {vrel[5, 5]}")
    # print(f"urel: {urel[5, 5]}")
    # print(f"wrel: {wrel[5, 5]}")
    # print(f"u: {u_new[5, 5]}")

    # print ("-------------------")
    # fl_kur = open("/Users/timurnezlobinskij/FrozenScience/Finitewave/examples/ikur.txt", "a")
    # fl_ical = open("/Users/timurnezlobinskij/FrozenScience/Finitewave/examples/ical.txt", "a")
    # fl_inaca = open("/Users/timurnezlobinskij/FrozenScience/Finitewave/examples/inaca.txt", "a")
    # fl_iks = open("/Users/timurnezlobinskij/FrozenScience/Finitewave/examples/iks.txt", "a")
    # fl_ik1 = open("/Users/timurnezlobinskij/FrozenScience/Finitewave/examples/ik1.txt", "a")
    # fl_cai = open("/Users/timurnezlobinskij/FrozenScience/Finitewave/examples/cai.txt", "a")

    for ind in prange(indexes.shape[0]):
        ii = indexes[ind]
        i = int(ii/n_j)
        j = ii % n_j

        ena, ek, eca = calc_equilibrum_potentials(nai[i, j], nao, ki[i, j], ko, cai[i, j], cao, R, T, F)

        
        m[i, j] = calc_gating_m(m[i, j], u[i, j], dt)
        h[i, j] = calc_gating_h(h[i, j], u[i, j], dt)
        j_[i, j] = calc_gating_j(j_[i, j], u[i, j], dt)

        ina = calc_ina(u[i, j], m[i, j], h[i, j], j_[i, j], gna, ena)

        ik1 = calc_ik1(u[i, j], gk1, ek)

        ito, oa[i, j], oi[i, j] = calc_ito(u[i, j], dt, kq10, oa[i, j], oi[i, j], gto, ek)

        ikur, ua[i, j], ui[i, j] = calc_ikur(u[i, j], dt, kq10, ua[i, j], ui[i, j], ek)

        ikr, xr[i, j] = calc_ikr(u[i, j], dt, xr[i, j], ek)

        iks, xs[i, j] = calc_iks(u[i, j], dt, xs[i, j], gks, ek)

        ical, d[i, j], f[i, j], fca[i, j] = calc_ical(u[i, j], dt, d[i, j], f[i, j], cai[i, j], gcal, fca[i, j])

        inak = calc_inak(inakmax, nai[i, j], nao, ko, kmnai, kmko, F, u[i, j], R, T)
        inaca = calc_inaca(inacamax, nai[i, j], nao, cai[i, j], cao, kmnancx, kmcancx, ksatncx, F, u[i, j], R, T)

        ibca = calc_ibca(gcab, eca, u[i, j])

        ibna = calc_ibna(gnab, ena, u[i, j])

        ipca = calc_ipca(ipcamax, cai[i, j])

        irel[i, j], urel[i, j], vrel[i, j], wrel[i, j] = calc_irel(dt, urel[i, j], vrel[i, j], irel[i, j], wrel[i, j], ical, inaca, krel, carel[i, j], cai[i, j], u[i, j], F, Vrel)
        itr = calc_itr(caup[i, j], carel[i, j])
        iup = calc_iup(iupmax, cai[i, j], kup)
        iupleak = calc_iupleak(caup[i, j], caupmax, iupmax) 
        # cmdn = calc_cmdn(cmdnmax, kmcmdn, cai[i, j])
        # trpn = calc_trpn(trpnmax, kmtrpn, cai[i, j])
        # csqn = calc_csqn(csqnmax, kmcsqn, carel[i, j])

        caup[i, j] = calc_caup(caup[i, j], dt, iup, iupleak, itr, Vrel, vup)
        nai[i, j] = calc_nai(nai[i, j], dt, inak, inaca, ibna, ina, F, Vj)

        ki[i, j] = calc_ki(ki[i, j], dt, inak, ik1, ito, ikur, ikr, iks, ibk, F, Vj)
        cai[i, j] = calc_cai(cai[i, j], dt, inaca, ipca, ical, ibca, iup, iupleak, irel[i, j], Vrel, vup, trpnmax, kmtrpn, cmdnmax, kmcmdn, F, Vj)

        carel[i, j] = calc_carel(carel[i, j], dt, itr, irel[i, j], csqnmax, kmcsqn)

        # if (i == 5 and j == 5):
        #     print(f"End u: {u[5,5]}")


        # if (i == 5 and j == 8):
        #     print(f"nai: {nai[i, j]}")
        #     print(f"ki: {ki[i, j]}")
        #     print(f"cai: {cai[i, j]}")
        #     print(f"caup: {caup[i, j]}")
        #     print(f"carel: {carel[i, j]}")
        #     print(f"m: {m[i, j]}")
        #     print(f"h: {h[i, j]}")
        #     print(f"j: {j_[i, j]}")
        #     print(f"d: {d[i, j]}")
        #     print(f"f: {f[i, j]}")
        #     print(f"oa: {oa[i, j]}")
        #     print(f"oi: {oi[i, j]}")
        #     print(f"ua: {ua[i, j]}")
        #     print(f"ui: {ui[i, j]}")
        #     print(f"xs: {xs[i, j]}")
        #     print(f"xr: {xr[i, j]}")
        #     print(f"fca: {fca[i, j]}")
        #     print(f"irel: {irel[i, j]}")
        #     print(f"vrel: {vrel[i, j]}")
        #     print(f"urel: {urel[i, j]}")
        #     print(f"wrel: {wrel[i, j]}")

        #     print(f"ina: {ina}")
        #     print(f"ik1: {ik1}")
        #     print(f"ito: {ito}")
        #     print(f"ikur: {ikur}")
        #     print(f"ikr: {ikr}")
        #     print(f"iks: {iks}")
        #     print(f"ical: {ical}")
        #     print(f"ipca: {ipca}")
        #     print(f"inak: {inak}")
        #     print(f"inaca: {inaca}")
        #     print(f"ibna: {ibna}")
        #     print(f"ibca: {ibca}")

        #     print (f"u: {u[i, j]}")
        #     print ("Total current: ", ina + ik1 + ito + ikur + ikr + iks + ical + ipca + inak + inaca + ibna + ibca)

        u_new[i, j] -= dt * (ina + ik1 + ito + ikur + ikr + iks + ical + ipca + inak + inaca + ibna + ibca)

    #     if (i == 10 and j == 10):
    #         fl_kur.write(str(ikur) + "\n")
    #         fl_ical.write(str(ical) + "\n")
    #         fl_inaca.write(str(inaca) + "\n")
    #         fl_iks.write(str(iks) + "\n")
    #         fl_ik1.write(str(ik1) + "\n")
    #         fl_cai.write(str(cai[i, j]) + "\n")


    # fl_kur.close()
    # fl_ical.close()
    # fl_inaca.close()
    # fl_iks.close()
    # fl_ik1.close()
    # fl_cai.close()