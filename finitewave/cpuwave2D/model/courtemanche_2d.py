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
        self.state_vars = ["u", "nai", "ki", "cai", "caup", "carel", "m", "h", "j",
         "d", "f", "oa", "oi", "ua", "ui", "xr", "xs", "fca", "irel"]
        
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
        self.ua = 0.004966*np.ones(shape, dtype=self.npfloat)
        self.ui = 0.999988*np.ones(shape, dtype=self.npfloat)
        self.xs = 0.0187*np.ones(shape, dtype=self.npfloat)
        self.xr = 0.0000329*np.ones(shape, dtype=self.npfloat)
        self.fca = 0.775*np.ones(shape, dtype=self.npfloat)
        self.irel = 0*np.ones(shape, dtype=self.npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel function to update ionic currents and state
        variables
        """
        ionic_kernel_2d(self.u_new, self.u, self.nai, self.ki, self.cai, self.caup, self.carel, 
                        self.m, self.h, self.j_, self.d, self.f, self.oa, self.oi, self.ua, 
                        self.ui, self.xr, self.xs, self.fca, self.irel, 
                        self.cardiac_tissue.myo_indexes, self.dt)

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


def calc_nai(nai, dt, inak, inaca, ibna, ina, F, Vj):
    """
    Calculates the intracellular sodium concentration.
    """
    dnai = (-3*inak-3*inaca - ibna - ina)/(F*Vj)
    nai += dt*dnai
    return nai

def calc_ki(ki, dt, inak, ik1, ito, ikur, ikr, iks, ibk, F, Vj):
    """
    Calculates the intracellular potassium concentration.
    """
    dki = (2*inak - ik1 - ito - ikur - ikr - iks - ibk)/(F*Vj)
    ki += dt*dki
    return ki

def calc_cai(cai, dt, inaca, ipca, ical, ibca, iup, iupleak, irel, urel, vup, trpnmax, kmtrpn, cmdnmax, kmcmdn, F, Vj): 
    """
    Calculates the intracellular calcium concentration.
    """
    B1 = (2*inaca - ipca - ical - ibca)/(2*F*Vj) + (vup*(iupleak - iup) + irel*urel)/Vj
    B2 = 1 + (trpnmax*kmtrpn)/((cai + kmtrpn)**2) + (cmdnmax*kmcmdn)/((cai + kmcmdn)**2)
    dcai = B1/B2
    cai += dt*dcai
    return cai

def calc_caup(caup, dt, iup, iupleak, itr, vrel, vup):
    """
    Calculates the calcium concentration in the up compartment.
    """
    dcaup = iup - iupleak - itr*(vrel/vup)
    caup += dt*dcaup
    return caup

def calc_carel(caup, carel, dt, itr, irel, csqnmax, kmcsqn):
    """
    Calculates the calcium concentration in the release compartment.
    """
    dcarel = (itr - irel)*(1 + (csqnmax*kmcsqn)/((carel + kmcsqn)**2))**(-1)
    carel += dt*dcarel
    return carel

def calc_equilibrum_potentials(u, nai, nao, ki, ko, cai, cao, R, T, F):
    """
    Calculates the equilibrum potentials for the cell.
    """
    ena = (R*T/F)*np.log(nao/nai)
    ek = (R*T/F)*np.log(ko/ki)
    eca = (R*T/(2*F))*np.log(cao/cai)
    return ena, ek, eca

def calc_ina(u, m, h, j, gna, ena):
    """
    Calculates the fast sodium current.
    """
    ina = gna*m**3*h*j*(u - ena)
    return ina


def calc_gating_m(m, u, dt):
    """
    Calculates the gating variable m for the fast sodium current.
    """
    am = 0
    if u == -47.13:
        am = 3.2
    else:
        am = 0.32*(u + 47.13)/(1 - np.exp(-0.1*(u + 47.13)))
    bm = 0.08*np.exp(-u/11)
    m_inf = am/(am + bm)
    tau_m = 1/(am + bm)
    m = m_inf - (m_inf - m)*np.exp(-dt/tau_m)
    return m

def calc_gating_h(h, u, dt):
    """
    Calculates the gating variable h for the fast sodium current.
    """
    ah = 0
    if u < -40:
        ah = 0.135*np.exp((80 + u)/-6.8)
    bh = 3.56*np.exp(0.079*u) + 310000*np.exp(0.35*u)
    h_inf = ah/(ah + bh)
    tau_h = 1/(ah + bh)
    h = h_inf - (h_inf - h)*np.exp(-dt/tau_h)
    return h

def calc_gating_j(j, u, dt):
    """
    Calculates the gating variable j for the fast sodium current.
    """
    aj = 0
    if u < -40:
        aj = (-127140*np.exp(0.2444*u) - 0.00003474*np.exp(-0.04391*u))*(u + 37.78)/(1 + np.exp(0.311*(u + 79.23)))
    bj = 0.1212*np.exp(-0.01052*u)/(1 + np.exp(-0.1378*(u + 40.14)))
    j_inf = aj/(aj + bj)
    tau_j = 1/(aj + bj)
    j = j_inf - (j_inf - j)*np.exp(-dt/tau_j)
    return j

def calc_ik1(u, gk1, ek):
    """
    Calculates the time-independent potassium current.
    """
    ik1 = gk1*(u - ek)/(1 + np.exp(0.07*(u + 80)))
    return ik1

def calc_ito(u, dt, kq10, oa, oi, gto, ek):
    """
    Calculates the transient outward potassium current.
    """
    ao = 0.65/(np.exp(-(u + 10)/8.5) + np.exp(-(u - 30)/59.0))
    bo = 0.65/(2.5 + np.exp((u + 82)/17.0))

    tau_o = kq10/(ao + bo)
    o_inf = 1/(1 + np.exp(-(u + 20.47)/17.54))

    aoi = 1/(18.53 + np.exp((u + 113.7)/10.95))
    boi = 1/(35.56 + np.exp(-(u + 1.26)/7.44))

    tau_oi = kq10/(aoi + boi)
    oi_inf = 1/(1 + np.exp((u + 43.1)/5.3))

    oa = o_inf - (o_inf - oa)*np.exp(-dt/tau_o)
    oi = oi_inf - (oi_inf - oi)*np.exp(-dt/tau_oi)

    ito = gto*(oa**3)*oi*(u - ek)  

    return ito, oa, oi

def calc_ikur(u, dt, kq10, ua, ui, ek):
    """
    Calculates the ultra-rapid delayed rectifier potassium current.
    """
    gkur = 0.005 + 0.05/(1 + np.exp(-(u - 15)/13))

    aua = 0.65/(np.exp(-(u + 10)/8.5) + np.exp(-(u - 30)/59.0))
    bua = 0.65/(2.5 + np.exp((u + 82)/17.0))
    
    tau_u = kq10/(aua + bua)
    ua_inf = 1/(1 + np.exp(-(u + 30.3)/9.6))

    aui = 1/(21 + np.exp(-(u - 185)/28.0))
    bui = np.exp((u - 158)/16.0)

    tau_ui = kq10/(aui + bui)
    ui_inf = 1/(1 + np.exp((u - 99.45)/27.48))

    ua = ua_inf - (ua_inf - u)*np.exp(-dt/tau_u)
    ui = ui_inf - (ui_inf - u)*np.exp(-dt/tau_ui)

    ikur = gkur*(ua**3)*ui*(u - ek)

    return ikur, ua, ui

def calc_ikr(u, dt, xr, gkr, ek):
    """
    Calculates the rapid delayed rectifier potassium current.
    """
    axr = 0.0003*(u + 14.1)/(1 - np.exp(-(u + 14.1)/5))
    bxr =0.000073898*(u - 3.3328)/(np.exp((u - 3.3328)/5.1237) - 1)

    tau_xr = 1/(axr + bxr)
    xr_inf = 1/(1 + np.exp(-(u + 14.1)/6.5))

    xr = xr_inf - (xr_inf - xr)*np.exp(-dt/tau_xr)

    ikr = (gkr*xr*(u - ek))/(1 + np.exp((u + 15)/22.4))

    return ikr, xr

def calc_iks(u, dt, xs, gks, ek):
    """
    Calculates the slow delayed rectifier potassium current.
    """
    axs = 0.00004*(u - 19.9)/(1 - np.exp(-(u - 19.9)/17))
    bxs = 0.000035*(u - 19.9)/(np.exp((u - 19.9)/9) - 1)

    tau_xs = 1/(2*(axs + bxs))
    xs_inf = 1/np.sqrt(1 + np.exp(-(u - 19.9)/12.7))

    xs = xs_inf - (xs_inf - xs)*np.exp(-dt/tau_xs)

    iks = gks*(xs**2)*(u - ek)

    return iks, xs

def calc_ical(u, dt, d, f, cai, gcal, fca):
    """
    Calculates the L-type calcium current.
    """
    tau_d = (1 - np.exp((u + 10)/-6.24))/(0.035*(u + 10)*(1 + np.exp((u + 10)/-6.24)))
    d_inf = 1/(1 + np.exp(-(u + 10)/8))

    tau_f = 9/(0.0197*np.exp(-((0.0337*(u + 10))**2)) + 0.02)
    f_inf = 1/(1 + np.exp((u + 28)/6.9))

    tau_fca = 2
    fca_inf = 1/(1 + cai/0.00035)

    d = d_inf - (d_inf - d)*np.exp(-dt/tau_d)
    f = f_inf - (f_inf - f)*np.exp(-dt/tau_f)
    fca = fca_inf - (fca_inf - fca)*np.exp(-dt/tau_fca)

    ical = gcal*d*f*fca*(u - 65)

    return ical, d, f, fca

def calc_inak(inakmax, nai, ko, kmnai, kmko, F, V, R, T):
    """
    Calculates the sodium-potassium pump current.
    """
    s = (1/7.0)*(np.exp(nai/67.3) - 1)
    fnak = 1/(1 + 0.1245*np.exp(-0.1*(F*V)/(R*T)) + 0.0365*s*np.exp((F*V)/(R*T)))

    inak = inakmax*fnak*(1/(1 + (kmnai/nai)**1.5))*(ko/(ko + kmko))

    return inak

def calc_inaca(inacamax, nai, nao, cai, cao, kmnancx, kmcancx, ksatncx, F, V, R, T):
    """
    Calculates the sodium-calcium exchanger current.
    """
    gamma = 0.35
    inaca = inacamax*(np.exp(gamma*(F*V)/(R*T))*nai**3*cao - np.exp((gamma - 1)*(F*V)/(R*T))*nao**3*cai)/(1 + (kmnancx/nai)**3*(kmcancx/cao)*(1 + ksatncx*np.exp((gamma - 1)*(F*V)/(R*T))))

    return inaca

def calc_ibca(gcab, eca, u):
    """
    Calculates the background calcium current.
    """
    ibca = gcab*(u - eca)
    return ibca

def calc_ibna(gnab, ena, u):
    """
    Calculates the background sodium current.
    """
    ibna = gnab*(u - ena)
    return ibna

def calc_ipca(ipcamax, cai):
    """
    Calculates the sarcolemmal calcium pump current.
    """
    ipca = ipcamax*cai/(cai + 0.0005)
    return ipca

def calc_irel(dt, urel, vrel, irel, wrel, ical, inaca, krel, carel, cai, F): 
    """
    Calculates the calcium release from the JSR.
    """
    tau_u = 8

    Fn = 1e-12*vrel*irel - ((5*1e-13)/F)*(0.5*ical - 0.2*inaca) 

    u_inf = 1/(1 + np.exp(-(Fn - 3.4175e-13)/13.67e-16))

    tau_v = 1.91 + 2.09/(1 + np.exp(-(Fn - 3.4175e-13)/13.67e-16))
    v_inf = 1 - 1/(1 + np.exp(-(Fn - 6.835e-14)/13.67e-16))

    tau_w = 6*(1 - np.exp(-(cai - 7.9)/5.0))/((1 + 0.3*np.exp(-(cai - 7.9)/5.0))*(cai - 7.9))
    w_inf = 1 - 1/(1 + np.exp(-(cai - 40)/17.0))

    urel = u_inf - (u_inf - urel)*np.exp(-dt/tau_u)
    vrel = v_inf - (v_inf - vrel)*np.exp(-dt/tau_v)
    wrel = w_inf - (w_inf - wrel)*np.exp(-dt/tau_w)

    irel = krel*(urel**2)*vrel*wrel*(carel - cai)

    return irel, urel, vrel, wrel

def calc_itr(caup, carel):
    """
    Calculates the transfer of calcium from the NSR to the JSR.
    """
    tautr = 180
    itr = (caup - carel)/tautr
    return itr

def calc_iup(iupmax, cai, kup):
    """
    Calculates the uptake of calcium into the NSR.
    """
    iup = iupmax/(1 + (kup/cai))
    return iup

def calc_iupleak(caup, caupmax, iupmax):
    """
    Calculates the leak of calcium from the NSR.
    """
    iupleak = (caup/caupmax)*iupmax
    return iupleak

def calc_cmdn(cmdnmax, kmcmdn, cai):
    """
    Calculates the concentration of calmodulin.
    """
    cmdn = cmdnmax*cai/(cai + kmcmdn)
    return cmdn

def calc_trpn(trpnmax, kmtrpn, cai):
    """
    Calculates the concentration of troponin.
    """
    trpn = trpnmax*cai/(cai + kmtrpn)
    return trpn

def calc_csqn(csqnmax, kmcsqn, carel):
    """
    Calculates the concentration of calsequestrin.
    """
    csqn = csqnmax*carel/(carel + kmcsqn)
    return csqn



# tp06 epi kernel
# @njit(parallel=True)
def ionic_kernel_2d(u_new, u, nai, ki, cai, caup, carel, m, h, j_, d, f, oa, oi, ua, ui, xs, xr, fca, irel, indexes, dt):
    n_i = u.shape[0]
    n_j = u.shape[1]

    gna = 12.0
    gnab = 0.000674

    gk1 = 0.35
    gkr = 0.035
    gks = 0.0035
    gto = 0.294
    gcal = 0.1238
    gcab = 0.000674

    F = 96485.0
    T = 310.0
    R = 8314 

    l = 0.01       # (cm) 
    a = 0.0008     # (cm) 
    Vj = 1000*np.pi*((a)**2)*l # (uL)
    vup = Vj*0.06*0.92
    vrel = Vj*0.06*0.08

    ibk = 0.0
    cao = 1.8 # mM
    nao = 140 # mM
    ko  = 4.5 # mM

    caupmax = 0.005 # mM/ms
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
    inacamax = 1750
    inakmax = 1.0933 
    ipcamax = 0.275
    krel = 30

    iupmax = 0.005
    V = -80

    kq10 = 2

    urel = 0
    vrel = 0
    wrel = 0


    for ind in prange(indexes.shape[0]):
        ii = indexes[ind]
        i = int(ii/n_j)
        j = ii % n_j

        ena, ek, eca = calc_equilibrum_potentials(u[i, j], nai[i, j], nao, ki[i, j], ko, cai[i, j], cao, R, T, F)

        m[i, j] = calc_gating_m(m[i, j], u[i, j], dt)
        h[i, j] = calc_gating_h(h[i, j], u[i, j], dt)
        j_[i, j] = calc_gating_j(j_[i, j], u[i, j], dt)


        ina = calc_ina(u[i, j], m[i, j], h[i, j], j_[i, j], gna, ena)
        ik1 = calc_ik1(u[i, j], gk1, ek)
        ito, oa[i, j], oi[i, j] = calc_ito(u[i, j], dt, kq10, oa[i, j], oi[i, j], gto, ek)
        ikur, ua[i, j], ui[i, j] = calc_ikur(u[i, j], dt, kq10, ua[i, j], ui[i, j], ek)
        ikr, xr[i, j] = calc_ikr(u[i, j], dt, xr[i, j], gkr, ek)
        iks, xs[i, j] = calc_iks(u[i, j], dt, xs[i, j], gks, ek)
        ical, d[i, j], f[i, j], fca[i, j] = calc_ical(u[i, j], dt, d[i, j], f[i, j], cai[i, j], gcal, fca[i, j])
        inak = calc_inak(inakmax, nai[i, j], ko, kmnai, kmko, F, V, R, T)
        inaca = calc_inaca(inacamax, nai[i, j], nao, cai[i, j], cao, kmnancx, kmcancx, ksatncx, F, V, R, T)
        ibca = calc_ibca(gcab, eca, u[i, j])
        ibna = calc_ibna(gnab, ena, u[i, j])
        ipca = calc_ipca(ipcamax, cai[i, j])
        irel[i, j], urel, vrel, wrel = calc_irel(dt, urel, vrel, irel[i, j], wrel, ical, inaca, krel, carel[i, j], cai[i, j], F)
        itr = calc_itr(caup[i, j], carel[i, j])
        iup = calc_iup(iupmax, cai[i, j], kup)
        iupleak = calc_iupleak(caup[i, j], caupmax, iupmax) 
        cmdn = calc_cmdn(cmdnmax, kmcmdn, cai[i, j])
        trpn = calc_trpn(trpnmax, kmtrpn, cai[i, j])
        csqn = calc_csqn(csqnmax, kmcsqn, carel[i, j])

        

        # print (caup[i, j], dt, iup, iupleak, itr, urel, vup)
        caup[i, j] = calc_caup(caup[i, j], dt, iup, iupleak, itr, urel, vup)

        nai[i, j] = calc_nai(nai[i, j], dt, inak, inaca, ibna, ina, F, Vj)
        ki[i, j] = calc_ki(ki[i, j], dt, inak, ik1, ito, ikur, ikr, iks, ibk, F, Vj)
        cai[i, j] = calc_cai(cai[i, j], dt, inaca, ipca, ical, ibca, iup, iupleak, irel[i, j], urel, vup, trpnmax, kmtrpn, cmdnmax, kmcmdn, F, Vj)

        carel[i, j] = calc_carel(caup[i, j], carel[i, j], dt, itr, irel[i, j], csqnmax, kmcsqn)

        u_new[i, j] -= dt * (ina + ik1 + ito + ikur + ikr + iks + ical + ipca + inak + inaca + ibna + ibca)


