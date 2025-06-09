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

        # Model parameters
        self.gna  = 7.8
        self.gnab = 0.000674
        self.gk1  = 0.09
        self.gkr  = 0.0294
        self.gks  = 0.129
        self.gto  = 0.1652
        self.gcal = 0.1238
        self.gcab = 0.00113

        self.gkur_coeff = 1

        self.F = 96485.0
        self.T = 310.0
        self.R = 8314.0 

        self.Vc   = 20100
        self.Vj   = self.Vc*0.68      # (uL)
        self.Vup  = self.Vj*0.06*0.92
        self.Vrel = self.Vj*0.06*0.08

        self.ibk = 0.0
        self.cao = 1.8 # mM
        self.nao = 140 # mM
        self.ko  = 5.4 # mM

        self.caupmax =  15     # mM/ms
        self.kup     = 0.00092 # mM

        self.kmnai   = 10
        self.kmko    = 1.5
        self.kmnancx = 87.5
        self.kmcancx = 1.38
        self.ksatncx = 0.1

        self.kmcmdn = 0.00238
        self.kmtrpn = 0.0005
        self.kmcsqn = 0.8

        self.trpnmax  = 0.07 # mM
        self.cmdnmax  = 0.05 # mM
        self.csqnmax  = 10.0 # mM
        self.inacamax = 1600
        self.inakmax  = 0.6
        self.ipcamax  = 0.275
        self.krel     = 30

        self.iupmax = 0.005

        self.kq10 = 3

        # initial conditions
        self.init_u     = -84.5
        self.init_nai   = 11.2
        self.init_ki    = 139
        self.init_cai   = 0.000102
        self.init_caup  = 1.6
        self.init_carel = 1.1
        self.init_m     = 0.00291
        self.init_h     = 0.965
        self.init_j     = 0.978
        self.init_d     = 0.000137
        self.init_f     = 0.999837
        self.init_oa    = 0.000592
        self.init_oi    = 0.9992
        self.init_ua    = 0.003519
        self.init_ui    = 0.9987
        self.init_xs    = 0.0187
        self.init_xr    = 0.0000329
        self.init_fca   = 0.775
        self.init_irel  = 0
        self.init_vrel  = 1
        self.init_urel  = 0
        self.init_wrel  = 0.9

    def initialize(self):
        """
        Initializes the model's state variables and diffusion/ionic kernels.

        Sets up the initial values for membrane potential, ion concentrations,
        gating variables, and assigns the appropriate kernel functions.
        """
        super().initialize()
        shape = self.cardiac_tissue.mesh.shape

        self.u = self.init_u * np.ones(shape, dtype=self.npfloat)         # (mV)
        self.u_new = self.u.copy()                                        # (mV)
        self.nai = self.init_nai * np.ones(shape, dtype=self.npfloat)     # (mM)
        self.ki = self.init_ki * np.ones(shape, dtype=self.npfloat)       # (mM)
        self.cai = self.init_cai * np.ones(shape, dtype=self.npfloat)     # (mM)
        self.caup = self.init_caup * np.ones(shape, dtype=self.npfloat)   # (mM)
        self.carel = self.init_carel * np.ones(shape, dtype=self.npfloat) # (mM)
        self.m = self.init_m * np.ones(shape, dtype=self.npfloat)    
        self.h = self.init_h * np.ones(shape, dtype=self.npfloat)      
        self.j_ = self.init_j * np.ones(shape, dtype=self.npfloat)
        self.d = self.init_d * np.ones(shape, dtype=self.npfloat)
        self.f = self.init_f * np.ones(shape, dtype=self.npfloat)
        self.oa = self.init_oa * np.ones(shape, dtype=self.npfloat)
        self.oi = self.init_oi * np.ones(shape, dtype=self.npfloat)
        self.ua = self.init_ua * np.ones(shape, dtype=self.npfloat)
        self.ui = self.init_ui * np.ones(shape, dtype=self.npfloat)
        self.xs = self.init_xs * np.ones(shape, dtype=self.npfloat)
        self.xr =self.init_xr * np.ones(shape, dtype=self.npfloat)
        self.fca = self.init_fca * np.ones(shape, dtype=self.npfloat)
        self.irel = self.init_irel * np.ones(shape, dtype=self.npfloat)
        self.vrel = self.init_vrel * np.ones(shape, dtype=self.npfloat)
        self.urel = self.init_urel * np.ones(shape, dtype=self.npfloat)
        self.wrel = self.init_wrel * np.ones(shape, dtype=self.npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel function to update ionic currents and state
        variables
        """
        ionic_kernel_2d(self.u_new, self.u, self.nai, self.ki, self.cai, self.caup, self.carel, 
                        self.m, self.h, self.j_, self.d, self.f, self.oa, self.oi, self.ua, 
                        self.ui, self.xr, self.xs, self.fca, self.irel, self.vrel, self.urel, 
                        self.wrel, self.cardiac_tissue.myo_indexes, self.dt, 
                        self.gna, self.gnab, self.gk1, self.gkr, self.gks, self.gto, self.gcal,
                        self.gcab, self.gkur_coeff, self.F, self.T, self.R, self.Vc, self.Vj, self.Vup,
                        self.Vrel, self.ibk, self.cao, self.nao, self.ko, self.caupmax,
                        self.kup, self.kmnai, self.kmko, self.kmnancx, self.kmcancx,
                        self.ksatncx, self.kmcmdn, self.kmtrpn, self.kmcsqn, self.trpnmax,
                        self.cmdnmax, self.csqnmax, self.inacamax, self.inakmax,
                        self.ipcamax, self.krel, self.iupmax, self.kq10)

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
    if x > 500:
        x = 500
    elif x < -500:
        x = -500
    return np.exp(x)

@njit
def calc_gating_variable(x, x_inf, tau_x, dt):
    """
    Calculates the gating variable using the steady-state value and time constant.
    """
    return x_inf - (x_inf - x)*np.exp(-dt/tau_x)

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
def calc_cai(cai, dt, inaca, ipca, ical, ibca, iup, iupleak, irel, Vrel, Vup, trpnmax, kmtrpn, cmdnmax, kmcmdn, F, Vj): 
    """
    Calculates the intracellular calcium concentration.
    """
    B1 = (2*inaca - ipca - ical - ibca)/(2*F*Vj) + (Vup*(iupleak - iup) + irel*Vrel)/Vj
    B2 = 1 + (trpnmax*kmtrpn)/((cai + kmtrpn)**2) + (cmdnmax*kmcmdn)/((cai + kmcmdn)**2)
    dcai = B1/B2
    cai += dt*dcai
    # print ("CAI:", cai)
    return cai

@njit
def calc_caup(caup, dt, iup, iupleak, itr, Vrel, Vup):
    """
    Calculates the calcium concentration in the up compartment.
    """
    dcaup = iup - iupleak - itr*(Vrel/Vup)
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
    ena = (R*T/F)*np.log(nao/nai)

    ek = (R*T/F)*np.log(ko/ki)
    
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
        am = 0.32 * (u + 47.13) / (1 - np.exp(-0.1 * (u + 47.13)))

    bm = 0.08*np.exp(-u/11)
    m_inf = am/(am + bm)
    tau_m = 1/(am + bm)
    m = calc_gating_variable(m, m_inf, tau_m, dt)

    return m

@njit
def calc_gating_h(h, u, dt):
    """
    Calculates the gating variable h for the fast sodium current.
    """
    ah = 0.135*np.exp(-(80 + u)/6.8)
    bh = 3.56*np.exp(0.079*u) + 310000*np.exp(0.35*u)
    if u >= -40:
        ah = 0    
        bh = 1/(0.13*(1 + np.exp(-(u + 10.66)/11.1)))
    
    h_inf = ah/(ah + bh)
    tau_h = 1/(ah + bh)
    h = calc_gating_variable(h, h_inf, tau_h, dt)
    return h

@njit
def calc_gating_j(j, u, dt):
    """
    Calculates the gating variable j for the fast sodium current.
    """
    
    aj = (-127140*np.exp(0.2444*u) - 0.00003474*np.exp(-0.04391*u))*(u + 37.78)/(1 + np.exp(0.311*(u + 79.23)))
    bj = 0.1212*np.exp(-0.01052*u)/(1 + np.exp(-0.1378*(u + 40.14)))
    if u >= -40:
        aj = 0
        bj = 0.3*np.exp(-0.0000002535*u)/(1 + np.exp(-0.1*(u + 32)))
    j_inf = aj/(aj + bj)
    tau_j = 1/(aj + bj)
    j = j_inf - (j_inf - j)*np.exp(-dt/tau_j)
    return j

@njit
def calc_ik1(u, gk1, ek):
    """
    Calculates the time-independent potassium current.
    """
    ik1 = gk1*(u - ek)/(1 + np.exp(0.07*(u + 80)))
    return ik1

@njit
def calc_ito(u, dt, kq10, oa, oi, gto, ek):
    """
    Calculates the transient outward potassium current.
    """
    ao = 0.65/(np.exp(-(u + 10)/8.5) + np.exp(-(u - 30)/59.0))
    bo = 0.65/(2.5 + np.exp((u + 82)/17.0))

    tau_o = 1/(kq10*(ao + bo))
    o_inf = 1/(1 + np.exp(-(u + 20.47)/17.54))

    aoi = 1/(18.53 + np.exp((u + 113.7)/10.95))
    boi = 1/(35.56 + np.exp(-(u + 1.26)/7.44))

    tau_oi = 1/(kq10*(aoi + boi))
    oi_inf = 1/(1 + np.exp((u + 43.1)/5.3))

    oa = calc_gating_variable(oa, o_inf, tau_o, dt)
    oi = calc_gating_variable(oi, oi_inf, tau_oi, dt)

    ito = gto*(oa**3)*oi*(u - ek)  

    return ito, oa, oi

@njit
def calc_ikur(u, dt, kq10, ua, ui, ek, gkur_coeff):
    """
    Calculates the ultra-rapid delayed rectifier potassium current.
    """
    gkur = 0.005 + 0.05/(1 + np.exp(-(u - 15)/13.0))

    aua = 0.65/(np.exp(-(u + 10)/8.5) + np.exp(-(u - 30)/59.0))
    bua = 0.65/(2.5 + np.exp((u + 82)/17.0))
    tau_ua = 1/(kq10*(aua + bua))
    ua_inf = 1/(1 + np.exp(-(u + 30.3)/9.6))
    aui = 1/(21 + np.exp(-(u - 185)/28.0))
    bui = np.exp((u - 158)/16.0)

    tau_ui = 1/(kq10*(aui + bui))
    ui_inf = 1/(1 + np.exp((u - 99.45)/27.48))

    ua = calc_gating_variable(ua, ua_inf, tau_ua, dt)
    ui = calc_gating_variable(ui, ui_inf, tau_ui, dt)

    ikur = gkur_coeff*gkur*(ua**3)*ui*(u - ek)

    return ikur, ua, ui

@njit
def calc_ikr(u, dt, xr, gkr, ek):
    """
    Calculates the rapid delayed rectifier potassium current.
    """
    gkr = 0.0294 # * np.sqrt(ko / 5.4)
    axr = 0.0003*(u + 14.1)/(1 - np.exp(-(u + 14.1)/5))
    bxr = 0.000073898*(u - 3.3328)/(np.exp((u - 3.3328)/5.1237) - 1)

    tau_xr = 1/(axr + bxr)
    xr_inf = 1/(1 + np.exp(-(u + 14.1)/6.5))

    xr = calc_gating_variable(xr, xr_inf, tau_xr, dt)

    ikr = (gkr*xr*(u - ek))/(1 + np.exp((u + 15)/22.4))

    return ikr, xr

@njit
def calc_iks(u, dt, xs, gks, ek):
    """
    Calculates the slow delayed rectifier potassium current.
    """
    axs = 0.00004*(u - 19.9)/(1 - np.exp(-(u - 19.9)/17))
    bxs = 0.000035*(u - 19.9)/(np.exp((u - 19.9)/9) - 1)

    tau_xs = 1/(2*(axs + bxs))
    xs_inf = 1/np.sqrt(1 + np.exp(-(u - 19.9)/12.7))

    xs = calc_gating_variable(xs, xs_inf, tau_xs, dt)

    iks = gks*(xs**2)*(u - ek)

    return iks, xs

@njit
def calc_ical(u, dt, d, f, cai, gcal, fca, eca):
    """
    Calculates the L-type calcium current.
    """
    tau_d = (1 - np.exp(-(u + 10)/6.24))/(0.035*(u + 10)*(1 + np.exp(-(u + 10)/6.24)))
    d_inf = 1/(1 + np.exp(-(u + 10)/8.0))

    tau_f = 9/(0.0197*np.exp(-(0.0337**2)*((u + 10)**2)) + 0.02)
    f_inf = 1/(1 + np.exp((u + 28)/6.9))

    tau_fca = 2
    fca_inf = 1/(1 + cai/0.00035)

    d   = calc_gating_variable(d, d_inf, tau_d, dt)
    f   = calc_gating_variable(f, f_inf, tau_f, dt)
    fca = calc_gating_variable(fca, fca_inf, tau_fca, dt)

    ical = gcal*d*f*fca*(u - 65) 

    return ical, d, f, fca

@njit
def calc_inak(inakmax, nai, nao, ko, kmnai, kmko, F, u, R, T):
    """
    Calculates the sodium-potassium pump current.
    """
    s = (1/7.0)*(np.exp(nao/67.3) - 1)
    fnak = 1/(1 + 0.1245*np.exp(-0.1*(F*u)/(R*T)) + 0.0365*s*np.exp(-(F*u)/(R*T)))

    inak = inakmax*fnak*(1/(1 + (kmnai/nai)**1.5))*(ko/(ko + kmko))

    return inak

@njit
def calc_inaca(inacamax, nai, nao, cai, cao, kmnancx, kmcancx, ksatncx, F, u, R, T):
    """
    Calculates the sodium-calcium exchanger current.
    """
    gamma = 0.35

    # Exponential terms with clamping
    exp_term = np.exp(gamma * (F * u) / (R * T))
    exp_rev_term = np.exp((gamma - 1) * (F * u) / (R * T))

    # Numerator
    numerator = inacamax * (exp_term * nai**3 * cao - exp_rev_term * nao**3 * cai)

    # Denominator
    term1 = (kmnancx**3 + nao**3)  # (K_m,Na^3 + [Na+]_i^3)
    term2 = (kmcancx + cao)        # (K_m,Ca + [Ca2+]_o)
    term3 = (1 + ksatncx * exp_rev_term)  # (1 + k_sat * exp(...))

    denominator = term1 * term2 * term3

    # Calculate INaCa
    inaca = numerator / denominator

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

    u_inf = 1/(1 + np.exp(-(Fn - 3.4175e-13)/13.67e-16))

    tau_v = 1.91 + 2.09/(1 + np.exp(-(Fn - 3.4175e-13)/13.67e-16))
    v_inf = 1 - 1/(1 + np.exp(-(Fn - 6.835e-14)/13.67e-16))

    tau_w = 6 * (1 - np.exp(-(u - 7.9) / 5.0)) / ((1 + 0.3 * np.exp(-(u - 7.9) / 5.0)) * (u - 7.9))
    w_inf = 1 - 1/(1 + np.exp(-(u - 40)/17.0))

    urel = calc_gating_variable(urel, u_inf, tau_u, dt)
    vrel = calc_gating_variable(vrel, v_inf, tau_v, dt)
    wrel = calc_gating_variable(wrel, w_inf, tau_w, dt)

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
def ionic_kernel_2d(u_new, u, nai, ki, cai, caup, carel, m, h, j_, d, f, oa, oi, ua, ui, xs, xr, fca, irel, vrel, urel, wrel, indexes, dt, 
                    gna, gnab, gk1, gkr, gks, gto, gcal, gcab, gkur_coeff, F, T, R, Vc, Vj, Vup, Vrel, ibk, cao, nao, ko, caupmax, kup,
                    kmnai, kmko, kmnancx, kmcancx, ksatncx, kmcmdn, kmtrpn, kmcsqn, trpnmax, cmdnmax, csqnmax, inacamax,
                    inakmax, ipcamax, krel, iupmax, kq10):
    """
    Computes the ionic currents and updates the state variables in the 2D
    Courtemanche cardiac model.

    Parameters
    ----------
    u_new : np.ndarray
        Updated membrane potential values.
    u : np.ndarray
        Current membrane potential values.
    v : np.ndarray
        Recovery variable array.
    indexes : np.ndarray
        Array of indices where the kernel should be computed (``mesh == 1``).
    dt : float
        Time step for the simulation.
    """
                    
    n_i = u.shape[0]
    n_j = u.shape[1]

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

        ikur, ua[i, j], ui[i, j] = calc_ikur(u[i, j], dt, kq10, ua[i, j], ui[i, j], ek, gkur_coeff)

        ikr, xr[i, j] = calc_ikr(u[i, j], dt, xr[i, j], gkr, ek)

        iks, xs[i, j] = calc_iks(u[i, j], dt, xs[i, j], gks, ek)

        ical, d[i, j], f[i, j], fca[i, j] = calc_ical(u[i, j], dt, d[i, j], f[i, j], cai[i, j], gcal, fca[i, j], eca)

        inak = calc_inak(inakmax, nai[i, j], nao, ko, kmnai, kmko, F, u[i, j], R, T)
        inaca = calc_inaca(inacamax, nai[i, j], nao, cai[i, j], cao, kmnancx, kmcancx, ksatncx, F, u[i, j], R, T)

        ibca = calc_ibca(gcab, eca, u[i, j])

        ibna = calc_ibna(gnab, ena, u[i, j])

        ipca = calc_ipca(ipcamax, cai[i, j])

        irel[i, j], urel[i, j], vrel[i, j], wrel[i, j] = calc_irel(dt, urel[i, j], vrel[i, j], irel[i, j], wrel[i, j], ical, inaca, krel, carel[i, j], cai[i, j], u[i, j], F, Vrel)
        itr = calc_itr(caup[i, j], carel[i, j])
        iup = calc_iup(iupmax, cai[i, j], kup)
        iupleak = calc_iupleak(caup[i, j], caupmax, iupmax) 

        caup[i, j] = calc_caup(caup[i, j], dt, iup, iupleak, itr, Vrel, Vup)
        nai[i, j] = calc_nai(nai[i, j], dt, inak, inaca, ibna, ina, F, Vj)

        ki[i, j] = calc_ki(ki[i, j], dt, inak, ik1, ito, ikur, ikr, iks, ibk, F, Vj)
        cai[i, j] = calc_cai(cai[i, j], dt, inaca, ipca, ical, ibca, iup, iupleak, irel[i, j], Vrel, Vup, trpnmax, kmtrpn, cmdnmax, kmcmdn, F, Vj)

        carel[i, j] = calc_carel(carel[i, j], dt, itr, irel[i, j], csqnmax, kmcsqn)

        u_new[i, j] -= dt * (ina + ik1 + ito + ikur + ikr + iks + ical + ipca + inak + inaca + ibna + ibca)
