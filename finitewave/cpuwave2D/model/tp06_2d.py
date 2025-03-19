import numpy as np
from numba import njit, prange

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import (
    AsymmetricStencil2D
)
from finitewave.cpuwave2D.stencil.isotropic_stencil_2d import (
    IsotropicStencil2D
)


class TP062D(CardiacModel):
    """
    A class to represent the TP06 cardiac model in 2D.

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
        self.state_vars = ["u", "Cai", "CaSR", "CaSS", "Nai", "Ki",
                           "M_", "H_", "J_", "Xr1", "Xr2", "Xs", "R_",
                           "S_", "D_", "F_", "F2_", "FCass", "RR", "OO"]
        self.npfloat = 'float64'

        self.Ko = 5.4
        self.Cao = 2.0
        self.Nao = 140.0

        self.Vc = 0.016404
        self.Vsr = 0.001094
        self.Vss = 0.00005468

        self.Bufc = 0.2
        self.Kbufc = 0.001
        self.Bufsr = 10.
        self.Kbufsr = 0.3
        self.Bufss = 0.4
        self.Kbufss = 0.00025

        self.Vmaxup = 0.006375
        self.Kup = 0.00025
        self.Vrel = 0.102  # 40.8
        self.k1_ = 0.15
        self.k2_ = 0.045
        self.k3 = 0.060
        self.k4 = 0.005  # 0.000015
        self.EC = 1.5
        self.maxsr = 2.5
        self.minsr = 1.
        self.Vleak = 0.00036
        self.Vxfer = 0.0038

        self.R = 8314.472
        self.F = 96485.3415
        self.T = 310.0
        self.RTONF = 26.713760659695648

        self.CAPACITANCE = 0.185

        self.Gkr = 0.153

        self.pKNa = 0.03

        self.GK1 = 5.405

        self.GNa = 14.838

        self.GbNa = 0.00029

        self.KmK = 1.0
        self.KmNa = 40.0
        self.knak = 2.724

        self.GCaL = 0.00003980

        self.GbCa = 0.000592

        self.knaca = 1000
        self.KmNai = 87.5
        self.KmCa = 1.38
        self.ksat = 0.1
        self.n_ = 0.35

        self.GpCa = 0.1238
        self.KpCa = 0.0005

        self.GpK = 0.0146

        self.Gto = 0.294
        self.Gks = 0.392

    def initialize(self):
        """
        Initializes the model's state variables and diffusion/ionic kernels.

        Sets up the initial values for membrane potential, ion concentrations,
        gating variables, and assigns the appropriate kernel functions.
        """
        super().initialize()
        shape = self.cardiac_tissue.mesh.shape

        self.u = -84.5*np.ones(shape, dtype=self.npfloat)
        self.u_new = self.u.copy()
        self.Cai = 0.00007*np.ones(shape, dtype=self.npfloat)
        self.CaSR = 1.3*np.ones(shape, dtype=self.npfloat)
        self.CaSS = 0.00007*np.ones(shape, dtype=self.npfloat)
        self.Nai = 7.67*np.ones(shape, dtype=self.npfloat)
        self.Ki = 138.3*np.ones(shape, dtype=self.npfloat)
        self.M_ = np.zeros(shape, dtype=self.npfloat)
        self.H_ = 0.75*np.ones(shape, dtype=self.npfloat)
        self.J_ = 0.75*np.ones(shape, dtype=self.npfloat)
        self.Xr1 = np.zeros(shape, dtype=self.npfloat)
        self.Xr2 = np.ones(shape, dtype=self.npfloat)
        self.Xs = np.zeros(shape, dtype=self.npfloat)
        self.R_ = np.zeros(shape, dtype=self.npfloat)
        self.S_ = np.ones(shape, dtype=self.npfloat)
        self.D_ = np.zeros(shape, dtype=self.npfloat)
        self.F_ = np.ones(shape, dtype=self.npfloat)
        self.F2_ = np.ones(shape, dtype=self.npfloat)
        self.FCass = np.ones(shape, dtype=self.npfloat)
        self.RR = np.ones(shape, dtype=self.npfloat)
        self.OO = np.zeros(shape, dtype=self.npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel function to update ionic currents and state
        variables
        """
        ionic_kernel_2d(self.u_new, self.u, self.Cai, self.CaSR, self.CaSS,
                        self.Nai, self.Ki, self.M_, self.H_, self.J_, self.Xr1,
                        self.Xr2, self.Xs, self.R_, self.S_, self.D_, self.F_,
                        self.F2_, self.FCass, self.RR, self.OO,
                        self.cardiac_tissue.myo_indexes, self.dt,
                        self.Ko, self.Cao, self.Nao, self.Vc, self.Vsr, self.Vss, self.Bufc, self.Kbufc, self.Bufsr, self.Kbufsr,
                        self.Bufss, self.Kbufss, self.Vmaxup, self.Kup, self.Vrel, self.k1_, self.k2_, self.k3, self.k4, self.EC,
                        self.maxsr, self.minsr, self.Vleak, self.Vxfer, self.R, self.F, self.T, self.RTONF, self.CAPACITANCE,
                        self.Gkr, self.pKNa, self.GK1, self.GNa, self.GbNa, self.KmK, self.KmNa, self.knak, self.GCaL, self.GbCa,
                        self.knaca, self.KmNai, self.KmCa, self.ksat, self.n_, self.GpCa, self.KpCa, self.GpK, self.Gto, self.Gks)

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
def calc_ina(u, dt, m, h, j, Gna, Ena):
    """
    Calculates the fast sodium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    dt : float
        Time step for the simulation.
    m : np.ndarray
        Gating variable for sodium channels (activation).
    h : np.ndarray
        Gating variable for sodium channels (inactivation).
    j : np.ndarray
        Gating variable for sodium channels (inactivation).
    Gna : float
        Sodium conductance.
    Ena : float
        Sodium reversal potential.
    """

    alpha_m = 1./(1.+np.exp((-60.-u)/5.))
    beta_m = 0.1/(1.+np.exp((u+35.)/5.)) + \
        0.10/(1.+np.exp((u-50.)/200.))
    tau_m = alpha_m*beta_m
    m_inf = 1./((1.+np.exp((-56.86-u)/9.03))
                * (1.+np.exp((-56.86-u)/9.03)))

    alpha_h = 0.
    beta_h = 0.
    if u >= -40.:
        alpha_h = 0.
        beta_h = 0.77/(0.13*(1.+np.exp(-(u+10.66)/11.1)))
    else:
        alpha_h = 0.057*np.exp(-(u+80.)/6.8)
        beta_h = 2.7*np.exp(0.079*u)+(3.1e5)*np.exp(0.3485*u)

    tau_h = 1.0/(alpha_h + beta_h)

    h_inf = 1./((1.+np.exp((u+71.55)/7.43))
                * (1.+np.exp((u+71.55)/7.43)))

    alpha_j = 0.
    beta_j = 0.
    if u >= -40.:
        alpha_j = 0.
        beta_j = 0.6*np.exp((0.057)*u)/(1.+np.exp(-0.1*(u+32.)))
    else:
        alpha_j = ((-2.5428e4)*np.exp(0.2444*u)-(6.948e-6) *
                np.exp(-0.04391*u))*(u+37.78) /\
            (1.+np.exp(0.311*(u+79.23)))
        beta_j = 0.02424*np.exp(-0.01052*u) / \
            (1.+np.exp(-0.1378*(u+40.14)))

    tau_j = 1.0/(alpha_j + beta_j)

    j_inf = h_inf

    m = m_inf-(m_inf-m)*np.exp(-dt/tau_m)
    h = h_inf-(h_inf-h)*np.exp(-dt/tau_h)
    j = j_inf-(j_inf-j)*np.exp(-dt/tau_j)

    return Gna*m*m*m*h*j*(u-Ena), m, h, j

@njit
def calc_ical(u, dt, d, f, f2, fcass, cao, cass, Gcal, F, R, T):
    """
    Calculates the L-type calcium current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    dt : float
        Time step for the simulation.
    d : np.ndarray
        Gating variable for L-type calcium channels.
    f : np.ndarray
        Gating variable for calcium-dependent calcium channels.
    f2 : np.ndarray
        Secondary gating variable for calcium-dependent calcium channels.
    fcass : np.ndarray
        Gating variable for calcium-sensitive current.
    cao : float
        Extracellular calcium concentration.
    cass : np.ndarray
        Calcium concentration in the submembrane space.
    Gcal : float
        Calcium conductance.
    F : float
        Faraday's constant.
    R : float
        Ideal gas constant.
    T : float
    """

    d_inf = 1./(1.+np.exp((-8-u)/7.5))
    Ad = 1.4/(1.+np.exp((-35-u)/13))+0.25
    Bd = 1.4/(1.+np.exp((u+5)/5))
    Cd = 1./(1.+np.exp((50-u)/20))
    tau_d = Ad*Bd+Cd
    f_inf = 1./(1.+np.exp((u+20)/7))
    Af = 1102.5*np.exp(-(u+27)*(u+27)/225)
    Bf = 200./(1+np.exp((13-u)/10.))
    Cf = (180./(1+np.exp((u+30)/10)))+20
    tau_f = Af+Bf+Cf
    f2_inf = 0.67/(1.+np.exp((u+35)/7))+0.33
    Af2 = 600*np.exp(-(u+25)*(u+25)/170)
    Bf2 = 31/(1.+np.exp((25-u)/10))
    Cf2 = 16/(1.+np.exp((u+30)/10))
    tau_f2 = Af2+Bf2+Cf2
    fcass_inf = 0.6/(1+(cass/0.05)*(cass/0.05))+0.4
    tau_fcass = 80./(1+(cass/0.05)*(cass/0.05))+2.

    d = d_inf-(d_inf-d)*np.exp(-dt/tau_d)
    f = f_inf-(f_inf-f)*np.exp(-dt/tau_f)
    f2 = f2_inf-(f2_inf-f2)*np.exp(-dt/tau_f2)
    fcass = fcass_inf-(fcass_inf-fcass)*np.exp(-dt/tau_fcass)

    return Gcal*d*f*f2*fcass*4*(u-15)*(F*F/(R*T)) *\
        (0.25*np.exp(2*(u-15)*F/(R*T))*cass-cao) / \
        (np.exp(2*(u-15)*F/(R*T))-1.), d, f, f2, fcass

@njit
def calc_ito(u, dt, r, s, Ek, Gto):
    """
    Calculates the transient outward current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    dt : float
        Time step for the simulation.
    r : np.ndarray
        Gating variable for ryanodine receptors.
    s : np.ndarray
        Gating variable for calcium-sensitive current.
    ek : float
        Potassium reversal potential.
    """

    r_inf = 1./(1.+np.exp((20-u)/6.))
    s_inf = 1./(1.+np.exp((u+20)/5.))
    tau_r = 9.5*np.exp(-(u+40.)*(u+40.)/1800.)+0.8
    tau_s = 85.*np.exp(-(u+45.)*(u+45.)/320.) + \
        5./(1.+np.exp((u-20.)/5.))+3.

    s = s_inf-(s_inf-s)*np.exp(-dt/tau_s)
    r = r_inf-(r_inf-r)*np.exp(-dt/tau_r)

    return Gto*r*s*(u-Ek), r, s

@njit
def calc_ikr(u, dt, xr1, xr2, Ek, Gkr, ko):
    """
    Calculates the rapid delayed rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    dt : float
        Time step for the simulation.
    xr1 : np.ndarray
        Gating variable for rapid delayed rectifier potassium channels.
    xr2 : np.ndarray
        Gating variable for rapid delayed rectifier potassium channels.
    Ek : float
        Potassium reversal potential.
    Gkr : float
        Potassium conductance.
    """

    xr1_inf = 1./(1.+np.exp((-26.-u)/7.))
    axr1 = 450./(1.+np.exp((-45.-u)/10.))
    bxr1 = 6./(1.+np.exp((u-(-30.))/11.5))
    tau_xr1 = axr1*bxr1
    xr2_inf = 1./(1.+np.exp((u-(-88.))/24.))
    axr2 = 3./(1.+np.exp((-60.-u)/20.))
    bxr2 = 1.12/(1.+np.exp((u-60.)/20.))
    tau_xr2 = axr2*bxr2

    xr1 = xr1_inf-(xr1_inf-xr1)*np.exp(-dt/tau_xr1)
    xr2 = xr2_inf-(xr2_inf-xr2)*np.exp(-dt/tau_xr2)

    return Gkr*np.sqrt(ko/5.4)*xr1*xr2*(u-Ek), xr1, xr2

@njit
def calc_iks(u, dt, xs, Eks, Gks):
    xs_inf = 1./(1.+np.exp((-5.-u)/14.))
    Axs = (1400./(np.sqrt(1.+np.exp((5.-u)/6))))
    Bxs = (1./(1.+np.exp((u-35.)/15.)))
    tau_xs = Axs*Bxs+80
    xs_inf = 1./(1.+np.exp((-5.-u)/14.))

    xs = xs_inf-(xs_inf-xs)*np.exp(-dt/tau_xs)

    return Gks*xs*xs*(u-Eks), xs

@njit
def calc_ik1(u, Ek, Gk1):
    """
    Calculates the inward rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    Ek : float
        Potassium reversal potential.
    Gk1 : float
        Inward rectifier potassium conductance.
    """

    ak1 = 0.1/(1.+np.exp(0.06*(u-Ek-200)))
    bk1 = (3.*np.exp(0.0002*(u-Ek+100)) +
           np.exp(0.1*(u-Ek-10)))/(1.+np.exp(-0.5*(u-Ek)))
    rec_iK1 = ak1/(ak1+bk1)

    return Gk1*rec_iK1*(u-Ek)

@njit
def calc_inaca(u, nao, nai, cao, cai, KmNai, KmCa, knaca, ksat, n_, F, R, T):
    """
    Calculates the sodium-calcium exchanger current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    nai : np.ndarray
        Sodium ion concentration in the intracellular space.
    cao : float
        Extracellular calcium concentration.
    cai : np.ndarray
        Calcium concentration in the submembrane space.
    KmNai : float
        Michaelis constant for sodium.
    KmCa : float
        Michaelis constant for calcium.
    knaca : float
        Sodium-calcium exchanger conductance.
    ksat : float
        Saturation factor.
    n_ : float
        Exponent factor.
    """

    return knaca*(1./(KmNai*KmNai*KmNai+nao*nao*nao))*(1./(KmCa+cao)) *\
            (1./(1+ksat*np.exp((n_-1)*u*F/(R*T)))) *\
            (np.exp(n_*u*F/(R*T))*nai*nai*nai*cao -
                np.exp((n_-1)*u*F/(R*T))*nao*nao*nao*cai*2.5)

@njit
def calc_inak(u, nai, ko, KmK, KmNa, knak, F, R, T):
    """
    Calculates the sodium-potassium pump current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    dt : float
        Time step for the simulation.
    nai : np.ndarray
        Sodium ion concentration in the intracellular space.
    ko : float
        Potassium ion concentration in the extracellular space.
    Nao : float
        Sodium ion concentration in the extracellular space.
    KmK : float
        Michaelis constant for potassium.
    KmNa : float
        Michaelis constant for sodium.
    knak : float
        Sodium-potassium pump conductance.
    pKNa : float
        Permeability ratio of sodium to potassium.
    """

    rec_iNaK = (
        1./(1.+0.1245*np.exp(-0.1*u*F/(R*T))+0.0353*np.exp(-u*F/(R*T))))

    return knak*(ko/(ko+KmK))*(nai/(nai+KmNa))*rec_iNaK

@njit
def calc_ipca(cai, KpCa, GpCa):
    """
    Calculates the calcium pump current.

    Parameters
    ----------
    cai : np.ndarray
        Calcium concentration in the submembrane space.
    KpCa : float
        Michaelis constant for calcium pump.
    GpCa : float
        Calcium pump conductance.
    """

    return GpCa*cai/(KpCa+cai)

@njit
def calc_ipk(u, Ek, GpK):
    """
    Calculates the potassium pump current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    dt : float
        Time step for the simulation.
    Ek : float
        Potassium reversal potential.
    rec_ipK : float
        Activation factor for potassium pump.
    GpK : float
        Potassium pump conductance.
    """
    rec_ipK = 1./(1.+np.exp((25-u)/5.98))

    return GpK*rec_ipK*(u-Ek)

@njit
def calc_ibna(u, Ena, GbNa):
    """
    Calculates the background sodium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    dt : float
        Time step for the simulation.
    Ena : float
        Sodium reversal potential.
    GbNa : float
        Background sodium conductance.
    """

    return GbNa*(u-Ena)

@njit
def calc_ibca(u, Eca, GbCa):
    """
    Calculates the background calcium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    dt : float
        Time step for the simulation.
    Eca : float
        Calcium reversal potential.
    GbCa : float
        Background calcium conductance.
    """

    return GbCa*(u-Eca)

@njit
def calc_irel(dt, rr, oo, casr, cass, vrel, k1, k2, k3, k4, maxsr, minsr, EC):
    """
    Calculates the ryanodine receptor current.

    Parameters
    ----------
    dt : float
        Time step for the simulation.
    rr : np.ndarray
        Ryanodine receptor gating variable for calcium release.
    oo : np.ndarray
        Ryanodine receptor gating variable for calcium release.
    caSR : np.ndarray
        Calcium concentration in the sarcoplasmic reticulum.
    caSS : np.ndarray
        Calcium concentration in the submembrane space.
    Vrel : float
        Release rate of calcium from the sarcoplasmic reticulum.
    """

    kCaSR = maxsr-((maxsr-minsr)/(1+(EC/casr)*(EC/casr)))
    k1_ = k1/kCaSR
    k2_ = k2*kCaSR
    dRR = k4*(1-rr)-k2_*cass*rr
    rr += dt*dRR
    oo = k1_*cass*cass * rr/(k3+k1_*cass*cass)

    return vrel*oo*(casr-cass), rr, oo

@njit
def calc_ileak(casr, cai, vleak):
    """
    Calculates the calcium leak current.

    Parameters
    ----------
    caSR : np.ndarray
        Calcium concentration in the sarcoplasmic reticulum.
    cai : np.ndarray
        Calcium concentration in the submembrane space.
    Vleak : float
        Leak rate of calcium from the sarcoplasmic reticulum.
    """

    return vleak*(casr-cai)

@njit
def calc_iup(cai, vmaxup, Kup):
    """
    Calculates the calcium uptake current.

    Parameters
    ----------
    cai : np.ndarray
        Calcium concentration in the submembrane space.
    Vmaxup : float
        Uptake rate of calcium into the sarcoplasmic reticulum.
    Kup : float
        Michaelis constant for calcium uptake.
    """

    return vmaxup/(1.+((Kup*Kup)/(cai*cai)))

@njit
def calc_ixfer(cass, cai, vxfer):
    """
    Calculates the calcium transfer current.

    Parameters
    ----------
    cass : np.ndarray
        Calcium concentration in the submembrane space.
    cai : np.ndarray
        Calcium concentration in the submembrane space.
    Vxfer : float
        Transfer rate of calcium between the submembrane space and cytosol.
    """

    return vxfer*(cass-cai)

@njit
def calc_casr(dt, caSR, bufsr, Kbufsr, iup, irel, ileak):
    """
    Calculates the calcium concentration in the sarcoplasmic reticulum.

    Parameters
    ----------
    caSR : np.ndarray
        Calcium concentration in the sarcoplasmic reticulum.
    bufsr : float
        Buffering capacity of the sarcoplasmic reticulum.
    Kbufsr : float
        Buffering constant of the sarcoplasmic reticulum.
    iup : float
        Calcium uptake current.
    irel : float
        Calcium release current.
    Ileak : float
        Leak rate of calcium from the sarcoplasmic reticulum.
    """

    CaCSQN = bufsr*caSR/(caSR+Kbufsr)
    dCaSR = dt*(iup-irel-ileak)
    bjsr = bufsr-CaCSQN-dCaSR-caSR+Kbufsr
    cjsr = Kbufsr*(CaCSQN+dCaSR+caSR)
    return (np.sqrt(bjsr*bjsr+4*cjsr)-bjsr)/2

@njit
def calc_cass(dt, caSS, bufss, Kbufss, ixfer, irel, ical, capacitance, Vc, Vss, Vsr, inversevssF2):
    """
    Calculates the calcium concentration in the submembrane space.

    Parameters
    ----------
    caSS : np.ndarray
        Calcium concentration in the submembrane space.
    bufss : float
        Buffering capacity of the submembrane space.
    Kbufss : float
        Buffering constant of the submembrane space.
    ixfer : float
        Calcium transfer current.
    irel : float
        Calcium release current.
    ical : float
        L-type calcium current.
    capacitance : float
        Membrane capacitance.
    Vc : float
        Volume of the cytosol.
    Vss : float
        Volume of the submembrane space.
    Vsr : float
        Volume of the sarcoplasmic reticulum.
    inversevssF2 : float
        Inverse of the product of 2
        times the volume of the submembrane space and Faraday's constant.
    """

    CaSSBuf = bufss*caSS/(caSS+Kbufss)
    dCaSS = dt*(-ixfer*(Vc/Vss)+irel*(Vsr/Vss) +
                (-ical*inversevssF2*capacitance))
    bcss = bufss-CaSSBuf-dCaSS-caSS+Kbufss
    ccss = Kbufss*(CaSSBuf+dCaSS+caSS)
    return (np.sqrt(bcss*bcss+4*ccss)-bcss)/2

@njit
def calc_cai(dt, cai, bufc, Kbufc, ibca, ipca, inaca, iup, ileak, ixfer, capacitance, vsr, vc, inverseVcF2):
    """
    Calculates the calcium concentration in the cytosol.

    Parameters
    ----------
    cai : np.ndarray
        Calcium concentration in the cytosol.
    bufc : float
        Buffering capacity of the cytosol.
    Kbufc : float
        Buffering constant of the cytosol.
    ibca : float
        Background calcium current.
    ipca : float
        Calcium pump current.
    inaca : float
        Sodium-calcium exchanger current.
    iup : float
        Calcium uptake current.
    ileak : float
        Calcium leak current.
    ixfer : float
        Calcium transfer current.
    capacitance : float
        Membrane capacitance.
    vsr : float
        Volume of the sarcoplasmic reticulum.
    vc : float
        Volume of the cytosol.
    inverseVcF2 : float
        Inverse of the product of 2
        times the volume of the cytosol and Faraday's constant.
    """

    CaCBuf = bufc*cai/(cai+Kbufc)
    dCai = dt*((-(ibca+ipca-2*inaca)*inverseVcF2*capacitance) -
                   (iup-ileak)*(vsr/vc)+ixfer)
    bc = bufc-CaCBuf-dCai-cai+Kbufc
    cc = Kbufc*(CaCBuf+dCai+cai)
    return (np.sqrt(bc*bc+4*cc)-bc)/2, cai

@njit
def calc_nai(dt, ina, ibna, inak, inaca, capacitance, inverseVcF):
    """
    Calculates the sodium concentration in the cytosol.

    Parameters
    ----------
    ina : float
        Fast sodium current.
    ibna : float
        Background sodium current.
    inak : float
        Sodium-potassium pump current.
    inaca : float
        Sodium-calcium exchanger current.
    capacitance : float
        Membrane capacitance.
    inverseVcF : float
        Inverse of the product of the volume of the cytosol and Faraday's constant.
    """

    dNai = -(ina+ibna+3*inak+3*inaca)*inverseVcF*capacitance
    return dt*dNai
### !!!! nai += ...

@njit
def calc_ki(dt, ik1, ito, ikr, iks, inak, ipk, inverseVcF, capacitance):
    """
    Calculates the potassium concentration in the cytosol.

    Parameters
    ----------
    ik1 : float
        Inward rectifier potassium current.
    ito : float
        Transient outward current.
    ikr : float
        Rapid delayed rectifier potassium current.
    iks : float
        Slow delayed rectifier potassium current.
    inak : float
        Sodium-potassium pump current.
    ipk : float
        Potassium pump current.
    capacitance : float
        Membrane capacitance.
    inverseVcF : float
        Inverse of the product of the volume of the cytosol and Faraday's constant.
    """

    dKi = -(ik1+ito+ikr+iks-2*inak+ipk)*inverseVcF*capacitance
    return dt*dKi
### !!!! ki += ...

# tp06 epi kernel
@njit(parallel=True)
def ionic_kernel_2d(u_new, u, Cai, CaSR, CaSS, Nai, Ki, M_, H_, J_, Xr1, Xr2,
                    Xs, R_, S_, D_, F_, F2_, FCass, RR, OO, indexes, dt, 
                    Ko, Cao, Nao, Vc, Vsr, Vss, Bufc, Kbufc, Bufsr, Kbufsr,
                    Bufss, Kbufss, Vmaxup, Kup, Vrel, k1_, k2_, k3, k4, EC,
                    maxsr, minsr, Vleak, Vxfer, R, F, T, RTONF, CAPACITANCE,
                    Gkr, pKNa, GK1, GNa, GbNa, KmK, KmNa, knak, GCaL, GbCa,
                    knaca, KmNai, KmCa, ksat, n_, GpCa, KpCa, GpK, Gto, Gks):
    """
    Compute the ionic currents and update the state variables for the 2D TP06
    cardiac model.

    This function calculates the ionic currents based on the TP06 cardiac
    model, updates ion concentrations, and modifies gating variables in the
    2D grid. The calculations are performed in parallel to enhance performance.

    Parameters
    ----------
    u_new : numpy.ndarray
        Array to store the updated membrane potential values.
    u : numpy.ndarray
        Array of current membrane potential values.
    Cai : numpy.ndarray
        Array of calcium concentration in the cytosol.
    CaSR : numpy.ndarray
        Array of calcium concentration in the sarcoplasmic reticulum.
    CaSS : numpy.ndarray
        Array of calcium concentration in the submembrane space.
    Nai : numpy.ndarray
        Array of sodium ion concentration in the intracellular space.
    Ki : numpy.ndarray
        Array of potassium ion concentration in the intracellular space.
    M_ : numpy.ndarray
        Array of gating variable for sodium channels (activation).
    H_ : numpy.ndarray
        Array of gating variable for sodium channels (inactivation).
    J_ : numpy.ndarray
        Array of gating variable for sodium channels (inactivation).
    Xr1 : numpy.ndarray
        Array of gating variable for rapid delayed rectifier potassium
        channels.
    Xr2 : numpy.ndarray
        Array of gating variable for rapid delayed rectifier potassium
        channels.
    Xs : numpy.ndarray
        Array of gating variable for slow delayed rectifier potassium channels.
    R_ : numpy.ndarray
        Array of gating variable for ryanodine receptors.
    S_ : numpy.ndarray
        Array of gating variable for calcium-sensitive current.
    D_ : numpy.ndarray
        Array of gating variable for L-type calcium channels.
    F_ : numpy.ndarray
        Array of gating variable for calcium-dependent calcium channels.
    F2_ : numpy.ndarray
        Array of secondary gating variable for calcium-dependent calcium
        channels.
    FCass : numpy.ndarray
        Array of gating variable for calcium-sensitive current.
    RR : numpy.ndarray
        Array of ryanodine receptor gating variable for calcium release.
    OO : numpy.ndarray
        Array of ryanodine receptor gating variable for calcium release.
    indexes: numpy.ndarray
        Array of indexes where the kernel should be computed (``mesh == 1``).
    dt : float
        Time step for the simulation.

    Returns
    -------
    None
        The function updates the state variables in place. No return value is
        produced.
    """
    n_j = u.shape[1]

    inverseVcF2 = 1./(2*Vc*F)
    inverseVcF = 1./(Vc*F)
    inversevssF2 = 1./(2*Vss*F)

    for ind in prange(len(indexes)):
        ii = indexes[ind]
        i = int(ii/n_j)
        j = ii % n_j

        Ek = RTONF*(np.log((Ko/Ki[i, j])))
        Ena = RTONF*(np.log((Nao/Nai[i, j])))
        Eks = RTONF*(np.log((Ko+pKNa*Nao)/(Ki[i, j]+pKNa*Nai[i, j])))
        Eca = 0.5*RTONF*(np.log((Cao/Cai[i, j])))

        # Compute currents
        ina, M_[i, j], H_[i, j], J_[i, j] = calc_ina(u[i, j], dt, M_[i, j], H_[i, j], J_[i, j], GNa, Ena)
        ical, D_[i, j], F_[i, j], F2_[i, j], FCass[i, j] = calc_ical(u[i, j], dt, D_[i, j], F_[i, j], F2_[i, j], FCass[i, j], Cao, CaSS[i, j], GCaL, F, R, T)
        ito, R_[i, j], S_[i, j] = calc_ito(u[i, j], dt, R_[i, j], S_[i, j], Ek, Gto)
        ikr, Xr1[i, j], Xr2[i, j] = calc_ikr(u[i, j], dt, Xr1[i, j], Xr2[i, j], Ek, Gkr, Ko)
        iks, Xs[i, j] = calc_iks(u[i, j], dt, Xs[i, j], Eks, Gks)
        ik1 = calc_ik1(u[i, j], Ek, GK1)
        inaca = calc_inaca(u[i, j], Nao, Nai[i, j], Cao, Cai[i, j], KmNai, KmCa, knaca, ksat, n_, F, R, T) 
        inak = calc_inak(u[i, j], Nai[i, j], Ko, KmK, KmNa, knak, F, R, T)
        ipca = calc_ipca(Cai[i, j], KpCa, GpCa)
        ipk = calc_ipk(u[i, j], Ek, GpK)
        ibna = calc_ibna(u[i, j], Ena, GbNa)
        ibca = calc_ibca(u[i, j], Eca, GbCa)
        irel, RR[i, j], OO[i, j] = calc_irel(dt, RR[i, j], OO[i, j], CaSR[i, j], CaSS[i, j], Vrel, k1_, k2_, k3, k4, maxsr, minsr, EC)
        ileak = calc_ileak(CaSR[i, j], Cai[i, j], Vleak)
        iup = calc_iup(Cai[i, j], Vmaxup, Kup)
        ixfer = calc_ixfer(CaSS[i, j], Cai[i, j], Vxfer)

        # Compute concentrations
        CaSR[i, j] = calc_casr(dt, CaSR[i, j], Bufsr, Kbufsr, iup, irel, ileak)
        CaSS[i, j] = calc_cass(dt, CaSS[i, j], Bufss, Kbufss, ixfer, irel, ical, CAPACITANCE, Vc, Vss, Vsr, inversevssF2)
        Cai[i, j], Cai[i, j] = calc_cai(dt, Cai[i, j], Bufc, Kbufc, ibca, ipca, inaca, iup, ileak, ixfer, CAPACITANCE, Vsr, Vc, inverseVcF2)
        Nai[i, j] += calc_nai(dt, ina, ibna, inak, inaca, CAPACITANCE, inverseVcF)
        Ki[i, j] += calc_ki(dt, ik1, ito, ikr, iks, inak, ipk, inverseVcF, CAPACITANCE)

        # Update membrane potential
        u_new[i, j] -= dt * (ikr + iks + ik1 + ito + ina + ibna + ical + ibca + inak + inaca + ipca + ipk)
        
