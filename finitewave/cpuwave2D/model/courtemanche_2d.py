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
        self.state_vars = ["u", "nai", "nao", "ki", "ko", "cai", 
                            "cao", "m", "h", "j", "d", "f", "xs", 
                            "xr", "ato", "iito", "uakur", "uikur", 
                            "fca", "ireljsrol"]
        
        self.npfloat = 'float64'

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

        self.nai = 11.2*np.ones(shape, dtype=self.npfloat)
        self.nao = 140*np.ones(shape, dtype=self.npfloat)
        self.ki = 139*np.ones(shape, dtype=self.npfloat)
        self.ko = 4.5*np.ones(shape, dtype=self.npfloat)
        self.cai = 0.000102*np.ones(shape, dtype=self.npfloat)
        self.cao = 1.8*np.ones(shape, dtype=self.npfloat)
        self.m = 0.00291*np.ones(shape, dtype=self.npfloat)
        self.h = 0.965*np.ones(shape, dtype=self.npfloat)
        self.j = 0.978*np.ones(shape, dtype=self.npfloat)
        self.d = 0.000137*np.ones(shape, dtype=self.npfloat)
        self.f = 0.999837*np.ones(shape, dtype=self.npfloat)
        self.xs = 0.0187*np.ones(shape, dtype=self.npfloat)
        self.xr = 0.0000329*np.ones(shape, dtype=self.npfloat)
        self.ato = 0.0304*np.ones(shape, dtype=self.npfloat)
        self.iito = 0.999*np.ones(shape, dtype=self.npfloat)
        self.uakur = 0.00496*np.ones(shape, dtype=self.npfloat)
        self.uikur = 0.999*np.ones(shape, dtype=self.npfloat)
        self.fca = 0.775*np.ones(shape, dtype=self.npfloat)
        self.ireljsrol = 0*np.ones(shape, dtype=self.npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel function to update ionic currents and state
        variables
        """
        ionic_kernel_2d(self.u_new, self.u, self.Cai, self.CaSR, self.CaSS,
                        self.Nai, self.Ki, self.M_, self.H_, self.J_, self.Xr1,
                        self.Xr2, self.Xs, self.R_, self.S_, self.D_, self.F_,
                        self.F2_, self.FCass, self.RR, self.OO,
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


# Functions that describe the currents begin here
@njit
def comp_ina(v, nai, nao, dt, m, h, j, R, temp, frdy):
    gna = 7.8
    ena = ((R * temp) / frdy) * np.log(nao / nai)

    am = 0.32 * (v + 47.13) / (1 - np.exp(-0.1 * (v + 47.13)))
    bm = 0.08 * np.exp(-v / 11)

    if v < -40:
        ah = 0.135 * np.exp((80 + v) / -6.8)
        bh = 3.56 * np.exp(0.079 * v) + 310000 * np.exp(0.35 * v)
        aj = (-127140 * np.exp(0.2444 * v) - 0.00003474 * np.exp(-0.04391 * v)) * ((v + 37.78) / (1 + np.exp(0.311 * (v + 79.23))))
        bj = (0.1212 * np.exp(-0.01052 * v)) / (1 + np.exp(-0.1378 * (v + 40.14)))
    else:
        ah = 0.0
        bh = 1 / (0.13 * (1 + np.exp((v + 10.66) / -11.1)))
        aj = 0.0
        bj = (0.3 * np.exp(-0.0000002535 * v)) / (1 + np.exp(-0.1 * (v + 32)))

    h = ah / (ah + bh) - ((ah / (ah + bh)) - h) * np.exp(-dt / (1 / (ah + bh)))
    j = aj / (aj + bj) - ((aj / (aj + bj)) - j) * np.exp(-dt / (1 / (aj + bj)))
    m = am / (am + bm) - ((am / (am + bm)) - m) * np.exp(-dt / (1 / (am + bm)))

    ina = gna * m * m * m * h * j * (v - ena)
    return ina, m, h, j

@njit
def comp_ical(v, cai, dt, d, f, fca, gcalbar):
    dss = 1 / (1 + np.exp(-(v + 10) / 8))
    taud = (1 - np.exp((v + 10) / -6.24)) / (0.035 * (v + 10) * (1 + np.exp((v + 10) / -6.24)))

    fss = 1 / (1 + np.exp((v + 28) / 6.9))
    tauf = 9 / (0.0197 * np.exp(-((0.0337 * (v + 10)) ** 2)) + 0.02)

    fcass = 1 / (1 + cai / 0.00035)
    taufca = 2

    d = dss - (dss - d) * np.exp(-dt / taud)
    f = fss - (fss - f) * np.exp(-dt / tauf)
    fca = fcass - (fcass - fca) * np.exp(-dt / taufca)

    ibarca = gcalbar * (v - 65)
    ilca = d * f * fca * ibarca

    return ilca, d, f, fca

@njit
def comp_ikr(v, ko, ki, dt, xr, R, temp, frdy):
    gkr = 0.0294 * np.sqrt(ko / 5.4)
    ekr = ((R * temp) / frdy) * np.log(ko / ki)

    xrss = 1 / (1 + np.exp(-(v + 14.1) / 6.5))
    tauxr = 1 / (0.0003 * (v + 14.1) / (1 - np.exp(-(v + 14.1) / 5)) + 0.000073898 * (v - 3.3328) / (np.exp((v - 3.3328) / 5.1237) - 1))

    xr = xrss - (xrss - xr) * np.exp(-dt / tauxr)
    r = 1 / (1 + np.exp((v + 15) / 22.4))

    ikr = gkr * xr * r * (v - ekr)
    return ikr, xr

@njit
def comp_iks(v, ko, ki, dt, xs, R, temp, frdy):
    gks = 0.129
    eks = ((R * temp) / frdy) * np.log(ko / ki)
    tauxs = 0.5 / (0.00004 * (v - 19.9) / (1 - np.exp(-(v - 19.9) / 17)) + 0.000035 * (v - 19.9) / (np.exp((v - 19.9) / 9) - 1))
    xsss = 1 / np.sqrt(1 + np.exp(-(v - 19.9) / 12.7))
    xs = xsss - (xsss - xs) * np.exp(-dt / tauxs)

    iks = gks * xs * xs * (v - eks)
    return iks, xs

@njit
def comp_iki(v, ko, ki, R, temp, frdy, dt):
    gki = 0.09 * (ko / 5.4) ** 0.4
    eki = ((R * temp) / frdy) * np.log(ko / ki)

    kin = 1 / (1 + np.exp(0.07 * (v + 80)))
    iki = gki * kin * (v - eki)
    return iki

@njit
def comp_ikach(v, ko, ki, dt, yach, ach, R, temp, frdy):
    gkach = 0.135
    ekach = ((R * temp) / frdy) * np.log(ko / ki)
    alphayach = 1.232e-2 / (1 + 0.0042 / ach) + 0.0002475
    betayach = 0.01 * np.exp(0.0133 * (v + 40))
    tauyach = 1 / (alphayach + betayach)
    yachss = alphayach / (alphayach + betayach)

    yach = yachss - (yachss - yach) * np.exp(-dt / tauyach)
    ikach = gkach * yach * (v - ekach) / (1 + np.exp((v + 20) / 20))
    return ikach, yach

@njit
def comp_ikur(v, ko, ki, dt, uakur, uikur, R, temp, frdy):
    gkur = 0.005 + 0.05 / (1 + np.exp(-(v - 15) / 13))
    ekur = ((R * temp) / frdy) * np.log(ko / ki)
    alphauakur = 0.65 / (np.exp(-(v + 10) / 8.5) + np.exp(-(v - 30) / 59.0))
    betauakur = 0.65 / (2.5 + np.exp((v + 82) / 17.0))
    tauuakur = 1 / (3 * (alphauakur + betauakur))
    uakurss = 1 / (1 + np.exp(-(v + 30.3) / 9.6))
    alphauikur = 1 / (21 + np.exp(-(v - 185) / 28))
    betauikur = np.exp((v - 158) / 16)
    tauuikur = 1 / (3 * (alphauikur + betauikur))
    uikurss = 1 / (1 + np.exp((v - 99.45) / 27.48))

    uakur = uakurss - (uakurss - uakur) * np.exp(-dt / tauuakur)
    uikur = uikurss - (uikurss - uikur) * np.exp(-dt / tauuikur)

    ikur = gkur * uakur * uakur * uakur * uikur * (v - ekur)
    return ikur, uakur, uikur

@njit
def comp_ito(v, ko, ki, dt, ato, iito, R, temp, frdy):
    gito = 0.1652
    erevto = ((R * temp) / frdy) * np.log(ko / ki)

    alphaato = 0.65 / (np.exp(-(v + 10) / 8.5) + np.exp(-(v - 30) / 59))
    betaato = 0.65 / (2.5 + np.exp((v + 82) / 17))
    tauato = 1 / (3 * (alphaato + betaato))
    atoss = 1 / (1 + np.exp(-(v + 20.47) / 17.54))
    ato = atoss - (atoss - ato) * np.exp(-dt / tauato)

    alphaiito = 1 / (18.53 + np.exp((v + 113.7) / 10.95))
    betaiito = 1 / (35.56 + np.exp(-(v + 1.26) / 7.44))
    tauiito = 1 / (3 * (alphaiito + betaiito))
    iitoss = 1 / (1 + np.exp((v + 43.1) / 5.3))
    iito = iitoss - (iitoss - iito) * np.exp(-dt / tauiito)

    ito = gito * ato * ato * ato * iito * (v - erevto)
    return ito, ato, iito

@njit
def comp_inaca(v, nai, nao, cai, cao, R, temp, frdy, gammas, kmnancx, kmcancx, ksatncx):
    inaca = 1750 * (np.exp(gammas * frdy * v / (R * temp)) * nai * nai * nai * cao - np.exp((gammas - 1) * frdy * v / (R * temp)) * nao * nao * nao * cai) / ((kmnancx ** 3 + nao ** 3) * (kmcancx + cao) * (1 + ksatncx * np.exp((gammas - 1) * frdy * v / (R * temp))))
    return inaca

@njit
def comp_inak(v, nai, ko, nao, ibarnak, kmnai, kmko):
    sigma = (np.exp(nao / 67.3) - 1) / 7
    fnak = (v + 150) / (v + 200)
    inak = ibarnak * fnak * (1 / (1 + (kmnai / nai) ** 1.5)) * (ko / (ko + kmko))
    return inak

@njit
def comp_ipca(cai, ibarpca, kmpca):
    ipca = (ibarpca * cai) / (kmpca + cai)
    return ipca

@njit
def comp_icab(v, cai, cao, R, temp, frdy):
    gcab = 0.00113
    ecan = ((R * temp) / frdy) * np.log(cao / cai)
    icab = gcab * (v - ecan)
    return icab

@njit
def comp_inab(v, nai, nao, R, temp, frdy):
    gnab = 0.000674
    enan = ((R * temp) / frdy) * np.log(nao / nai)
    inab = gnab * (v - enan)
    return inab

@njit
def conc_nai(nai, naiont, acap, vmyo, zna, frdy, dt):
    dnai = -dt * naiont * acap / (vmyo * zna * frdy)
    nai += dnai
    return nai

@njit
def conc_ki(ki, kiont, acap, vmyo, zk, frdy, dt):
    dki = -dt * kiont * acap / (vmyo * zk * frdy)
    ki += dki
    return ki

@njit
def conc_nsr(nsr, iup, ileak, itr, iupbar, nsrbar, cai, kmup, vjsr, vnsr, dt):
    kleak = iupbar / nsrbar
    ileak = kleak * nsr

    iup = iupbar * cai / (cai + kmup)
    dnsr = dt * (iup - ileak - itr * vjsr / vnsr)
    nsr += dnsr
    return nsr, ileak, iup

@njit
def conc_jsr(jsr, itr, ireljsrol, vjsr, caiont, acap, frdy, grelbarjsrol, cai, v, csqnbar, kmcsqn, dt):
    fn = vjsr * (1e-12) * ireljsrol - (1e-12) * caiont * acap / (2 * frdy)

    tauurel = 8.0
    urelss = 1 / (1 + np.exp(-(fn - 3.4175e-13) / 13.67e-16))
    tauvrel = 1.91 + 2.09 / (1 + np.exp(-(fn - 3.4175e-13) / 13.67e-16))
    vrelss = 1 - 1 / (1 + np.exp(-(fn - 6.835e-14) / 13.67e-16))
    tauwrel = 6.0 * (1 - np.exp(-(v - 7.9) / 5)) / ((1 + 0.3 * np.exp(-(v - 7.9) / 5)) * (v - 7.9))
    wrelss = 1 - 1 / (1 + np.exp(-(v - 40) / 17))

    urel = urelss - (urelss - urel) * np.exp(-dt / tauurel)
    vrel = vrelss - (vrelss - vrel) * np.exp(-dt / tauvrel)
    wrel = wrelss - (wrelss - wrel) * np.exp(-dt / tauwrel)

    greljsrol = grelbarjsrol * urel * urel * vrel * wrel
    ireljsrol = greljsrol * (jsr - cai)

    djsr = dt * (itr - 0.5 * ireljsrol) / (1 + csqnbar * kmcsqn / ((jsr + kmcsqn) ** 2))
    jsr += djsr
    return jsr, urel, vrel, wrel

@njit
def calc_itr(nsr, jsr, tautr):
    itr = (nsr - jsr) / tautr
    return itr

@njit
def conc_cai(cai, caiont, dt, ileak, iup, ireljsrol, trpnbar, kmtrpn, cmdnbar, kmcmdn, acap, frdy, vmyo, vnsr, vjsr):
    trpn = trpnbar * (cai / (cai + kmtrpn))
    cmdn = cmdnbar * (cai / (cai + kmcmdn))

    b1cai = -caiont * acap / (2 * frdy * vmyo) + (vnsr * (ileak - iup) + 0.5 * ireljsrol * vjsr) / vmyo
    b2cai = 1 + trpnbar * kmtrpn / ((cai + kmtrpn) ** 2) + cmdn * kmcmdn / ((cai + kmcmdn) ** 2)
    dcai = dt * b1cai / b2cai

    cai += dcai
    return cai

# tp06 epi kernel
@njit(parallel=True)
def ionic_kernel_2d(u_new, u, nai, nao, ki, ko, cai, cao, m, h, j, d, f, xs, xr, ato, iito, uakur, uikur, fca, ireljsrol, indexes, dt):
    n_i = u.shape[0]
    n_j = u.shape[1]

    # Cell Geometry
    l = 0.01       # Length of the cell (cm) 
    a = 0.0008     # Radius of the cell (cm) 
    pi = 3.141592  # Pi 

    # Terms for Solution of Conductance and Reversal Potential */
    R = 8314      # Universal Gas Constant (J/kmol*K) */
    frdy = 96485  # Faraday's Constant (C/mol) */
    temp = 310    # Temperature (K) */

    # Ion Valences */
    zna = 1  # Na valence */
    zk = 1   # K valence */
    zca = 2  # Ca valence */

    # NSR Ca Ion Concentration Changes */
    kmup = 0.00092    # Half-saturation concentration of iup (mM) */
    iupbar = 0.005  # Max. current through iup channel (mM/ms) */
    nsrbar = 15       # Max. [Ca] in NSR (mM) */

    # JSR Ca Ion Concentration Changes */
    grelbarjsrol = 30 # Rate constant of Ca release from JSR due to overload (ms^-1)*/
    csqnbar = 10      # Max. [Ca] buffered in CSQN (mM)*/
    kmcsqn = 0.8      # Equalibrium constant of buffering for CSQN (mM)*/

    # Translocation of Ca Ions from NSR to JSR */
    tautr = 180  # Time constant of Ca transfer from NSR to JSR (ms)*/

    # Myoplasmic Ca Ion Concentration Changes */
    cmdnbar = 0.050   # Max. [Ca] buffered in CMDN (mM) */
    trpnbar = 0.070   # Max. [Ca] buffered in TRPN (mM) */
    kmcmdn = 0.00238  # Equalibrium constant of buffering for CMDN (mM) */
    kmtrpn = 0.0005   # Equalibrium constant of buffering for TRPN (mM) */

    gcalbar = 0.1238

    ach = 0.0 # Acetylcholine concentration */

    prnak = 0.01833  # Na/K Permiability Ratio */

    # Sodium-Calcium Exchanger */
    kmnancx = 87.5  # Na saturation constant for NaCa exchanger */
    ksatncx = 0.1   # Saturation factor for NaCa exchanger */
    kmcancx = 1.38  # Ca saturation factor for NaCa exchanger */
    gammas = 0.35  # Position of energy barrier controlling voltage dependance of inaca */

    # Sodium-Potassium Pump */
    ibarnak = 1.0933   # Max. current through Na-K pump (uA/uF) */
    kmnai = 10    # Half-saturation concentration of NaK pump (mM) */
    kmko = 1.5    # Half-saturation concentration of NaK pump (mM) */
        
    # Sarcolemmal Ca Pump */
    ibarpca = 0.275 # Max. Ca current through sarcolemmal Ca pump (uA/uF) */
    kmpca = 0.0005 # Half-saturation concentration of sarcolemmal Ca pump (mM) */

    # Cell Geometry
    vcell = 1000 * pi * a * a * l  # 3.801e-5 uL
    ageo = 2 * pi * a * a + 2 * pi * a * l
    acap = ageo * 2  # 1.534e-4 cm^2
    vmyo = vcell * 0.68
    vmito = vcell * 0.26
    vsr = vcell * 0.06
    vnsr = vcell * 0.0552
    vjsr = vcell * 0.0048

    # Initial Conditions
    jsr = 1.49
    nsr = 1.49
    yach = 2.54e-2

    for ind in prange(indexes.shape[0]):
        ii = indexes[ind]
        i = int(ii/n_j)
        j = ii % n_j

        # List of functions called for each timestep, currents commented out are only used when modeling pathological conditions */
        ina, m, h, j = comp_ina(u[i, j], nai, nao, dt, m, h, j, R, temp, frdy)
        ilca, d, f, fca = comp_ical(u[i, j], cai, dt, d, f, fca, gcalbar)
        ikr, xr = comp_ikr(u[i, j], ko, ki, dt, xr, R, temp, frdy)
        iks, xs = comp_iks(u[i, j], ko, ki, dt, xs, R, temp, frdy)
        iki = comp_iki(u[i, j], ko, ki, R, temp, frdy, dt)
        ikach, yach = comp_ikach(u[i, j], ko, ki, dt, yach, ach, R, temp, frdy)
        ikur, uakur, uikur = comp_ikur(u[i, j], ko, ki, dt, uakur, uikur, R, temp, frdy)
        ito, ato, iito = comp_ito(u[i, j], ko, ki, dt, ato, iito, R, temp, frdy)
        inaca = comp_inaca(u[i, j], nai, nao, cai, cao, R, temp, frdy, gammas, kmnancx, kmcancx, ksatncx)
        inak = comp_inak(u[i, j], nai, ko, nao, ibarnak, kmnai, kmko)
        ipca = comp_ipca(cai, ibarpca, kmpca)
        icab = comp_icab(u[i, j], cai, cao, R, temp, frdy)
        inab = comp_inab(u[i, j], nai, nao, R, temp, frdy)

        caiont = ilca+icab+ipca-2*inaca

        nai = conc_nai(nai, ina + inab + 3 * inak + 3 * inaca, acap, vmyo, zna, frdy, dt)
        ki = conc_ki(ki, iki + ikr + iks + ikach + ikur + ito - 2 * inak, acap, vmyo, zk, frdy, dt)
        nsr, ileak, iup = conc_nsr(nsr, itr, iupbar, nsrbar, cai, kmup, vjsr, vnsr, dt)
        jsr, urel, vrel, wrel = conc_jsr(jsr, itr, ireljsrol, vjsr, caiont, acap, frdy, grelbarjsrol, cai, u[i, j], csqnbar, kmcsqn, dt)
        itr = calc_itr(nsr, jsr, tautr)
        cai = conc_cai(cai, caiont, dt, ileak, iup, ireljsrol, trpnbar, kmtrpn, cmdnbar, kmcmdn, acap, frdy, vmyo, vnsr, vjsr)

        u_new[i, j] = u[i, j] + dt * (ina + ilca + ikr + iks + iki + ikach + ikur + ito + inaca + inak + ipca + icab + inab)
   


    