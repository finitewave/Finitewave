import numpy as np
from numba import njit, prange

from finitewave.cpuwave2D.model.tp06_2d import (
    TP062D,
    calc_ina,
    calc_ical,
    calc_ito,
    calc_ikr,
    calc_iks,
    calc_ik1,
    calc_inaca,
    calc_inak,
    calc_ipca,
    calc_ipk,
    calc_ibna,
    calc_ibca,
    calc_irel,
    calc_ileak,
    calc_iup,
    calc_ixfer,
    calc_casr,
    calc_cass,
    calc_cai,
    calc_nai,
    calc_ki
)
from finitewave.cpuwave3D.stencil.isotropic_stencil_3d import (
    IsotropicStencil3D
)
from finitewave.cpuwave3D.stencil.asymmetric_stencil_3d import (
    AsymmetricStencil3D
)


class TP063D(TP062D):
    """
    A class to represent the TP06 cardiac model in 3D.
    """

    def __init__(self):
        super().__init__()

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel function to update ionic currents and state
        variables.
        """
        ionic_kernel_3d(self.u_new, self.u, self.cai, self.casr, self.cass,
                        self.nai, self.Ki, self.m, self.h, self.j, self.xr1,
                        self.xr2, self.xs, self.r, self.s, self.d, self.f,
                        self.f2, self.fcass, self.rr, self.oo,
                        self.cardiac_tissue.myo_indexes, self.dt,
                        self.ko, self.cao, self.nao, self.Vc, self.Vsr, self.Vss, self.Bufc, self.Kbufc, self.Bufsr, self.Kbufsr,
                        self.Bufss, self.Kbufss, self.Vmaxup, self.Kup, self.Vrel, self.k1_, self.k2_, self.k3, self.k4, self.EC,
                        self.maxsr, self.minsr, self.Vleak, self.Vxfer, self.R, self.F, self.T, self.RTONF, self.CAPACITANCE,
                        self.gkr, self.pKNa, self.gk1, self.gna, self.gbna, self.KmK, self.KmNa, self.knak, self.gcal, self.gbca,
                        self.knaca, self.KmNai, self.KmCa, self.ksat, self.n_, self.gpca, self.KpCa, self.gpk, self.gto, self.gks)

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
            return IsotropicStencil3D()

        return AsymmetricStencil3D()


# tp06 epi kernel
@njit(parallel=True)
def ionic_kernel_3d(u_new, u, cai, casr, cass, nai, Ki, m, h, j_, xr1, xr2,
                    xs, r, s, d, f, f2, fcass, rr, oo, indexes, dt, 
                    ko, cao, nao, Vc, Vsr, Vss, Bufc, Kbufc, Bufsr, Kbufsr,
                    Bufss, Kbufss, Vmaxup, Kup, Vrel, k1_, k2_, k3, k4, EC,
                    maxsr, minsr, Vleak, Vxfer, R, F, T, RTONF, CAPACITANCE,
                    gkr, pKNa, gk1, gna, gbna, KmK, KmNa, knak, gcal, gbca,
                    knaca, KmNai, KmCa, ksat, n_, gpca, KpCa, gpk, gto, gks):
    """
    Compute the ionic currents and update the state variables for the 3D TP06
    cardiac model.

    This function calculates the ionic currents based on the TP06 cardiac
    model, updates ion concentrations, and modifies gating variables in
    the 3D grid. The calculations are performed in parallel to enhance
    performance.

    Parameters
    ----------
    u_new : numpy.ndarray
        Array to store the updated membrane potential values.
    u : numpy.ndarray
        Array of current membrane potential values.
    cai : numpy.ndarray
        Array of calcium concentration in the cytosol.
    casr : numpy.ndarray
        Array of calcium concentration in the sarcoplasmic reticulum.
    cass : numpy.ndarray
        Array of calcium concentration in the submembrane space.
    nai : numpy.ndarray
        Array of sodium ion concentration in the intracellular space.
    ki : numpy.ndarray
        Array of potassium ion concentration in the intracellular space.
    m : numpy.ndarray
        Array of gating variable for sodium channels (activation).
    h : numpy.ndarray
        Array of gating variable for sodium channels (inactivation).
    j_ : numpy.ndarray
        Array of gating variable for sodium channels (inactivation).
    xr1 : numpy.ndarray
        Array of gating variable for rapid delayed rectifier potassium
        channels.
    xr2 : numpy.ndarray
        Array of gating variable for rapid delayed rectifier potassium
        channels.
    xs : numpy.ndarray
        Array of gating variable for slow delayed rectifier potassium channels.
    r : numpy.ndarray
        Array of gating variable for ryanodine receptors.
    s : numpy.ndarray
        Array of gating variable for calcium-sensitive current.
    d : numpy.ndarray
        Array of gating variable for L-type calcium channels.
    f : numpy.ndarray
        Array of gating variable for calcium-dependent calcium channels.
    f2 : numpy.ndarray
        Array of secondary gating variable for calcium-dependent calcium
        channels.
    fcass : numpy.ndarray
        Array of gating variable for calcium-sensitive current.
    rr : numpy.ndarray
        Array of ryanodine receptor gating variable for calcium release.
    oo : numpy.ndarray
        Array of ryanodine receptor gating variable for calcium release.
    indexes : numpy.ndarray
        Array of indices where the kernel should be computed (``mesh == 1``).
    dt : float
        Time step for the simulation.

    Returns
    -------
    None
        The function updates the state variables in place. No return value is
        produced.
    """
    n_j = u.shape[1]
    n_k = u.shape[2]

    inverseVcF2 = 1./(2*Vc*F)
    inverseVcF = 1./(Vc*F)
    inversevssF2 = 1./(2*Vss*F)

    for ind in prange(len(indexes)):
        ii = indexes[ind]
        i = ii//(n_j*n_k)
        j = (ii % (n_j*n_k))//n_k
        k = (ii % (n_j*n_k)) % n_k

        Ek = RTONF*(np.log((ko/Ki[i, j, k])))
        Ena = RTONF*(np.log((nao/nai[i, j, k])))
        Eks = RTONF*(np.log((ko+pKNa*nao)/(Ki[i, j, k]+pKNa*nai[i, j, k])))
        Eca = 0.5*RTONF*(np.log((cao/cai[i, j, k])))

        # Compute currents
        ina, m[i, j, k], h[i, j, k], j_[i, j, k] = calc_ina(u[i, j, k], dt, m[i, j, k], h[i, j, k], j_[i, j, k], gna, Ena)
        ical, d[i, j, k], f[i, j, k], f2[i, j, k], fcass[i, j, k] = calc_ical(u[i, j, k], dt, d[i, j, k], f[i, j, k], f2[i, j, k], fcass[i, j, k], cao, cass[i, j, k], gcal, F, R, T)
        ito, r[i, j, k], s[i, j, k] = calc_ito(u[i, j, k], dt, r[i, j, k], s[i, j, k], Ek, gto)
        ikr, xr1[i, j, k], xr2[i, j, k] = calc_ikr(u[i, j, k], dt, xr1[i, j, k], xr2[i, j, k], Ek, gkr, ko)
        iks, xs[i, j, k] = calc_iks(u[i, j, k], dt, xs[i, j, k], Eks, gks)
        ik1 = calc_ik1(u[i, j, k], Ek, gk1)
        inaca = calc_inaca(u[i, j, k], nao, nai[i, j, k], cao, cai[i, j, k], KmNai, KmCa, knaca, ksat, n_, F, R, T) 
        inak = calc_inak(u[i, j, k], nai[i, j, k], ko, KmK, KmNa, knak, F, R, T)
        ipca = calc_ipca(cai[i, j, k], KpCa, gpca)
        ipk = calc_ipk(u[i, j, k], Ek, gpk)
        ibna = calc_ibna(u[i, j, k], Ena, gbna)
        ibca = calc_ibca(u[i, j, k], Eca, gbca)
        irel, rr[i, j, k], oo[i, j, k] = calc_irel(dt, rr[i, j, k], oo[i, j, k], casr[i, j, k], cass[i, j, k], Vrel, k1_, k2_, k3, k4, maxsr, minsr, EC)
        ileak = calc_ileak(casr[i, j, k], cai[i, j, k], Vleak)
        iup = calc_iup(cai[i, j, k], Vmaxup, Kup)
        ixfer = calc_ixfer(cass[i, j, k], cai[i, j, k], Vxfer)

        # Compute concentrations
        casr[i, j, k] = calc_casr(dt, casr[i, j, k], Bufsr, Kbufsr, iup, irel, ileak)
        cass[i, j, k] = calc_cass(dt, cass[i, j, k], Bufss, Kbufss, ixfer, irel, ical, CAPACITANCE, Vc, Vss, Vsr, inversevssF2)
        cai[i, j, k], cai[i, j, k] = calc_cai(dt, cai[i, j, k], Bufc, Kbufc, ibca, ipca, inaca, iup, ileak, ixfer, CAPACITANCE, Vsr, Vc, inverseVcF2)
        nai[i, j, k] += calc_nai(dt, ina, ibna, inak, inaca, CAPACITANCE, inverseVcF)
        Ki[i, j, k] += calc_ki(dt, ik1, ito, ikr, iks, inak, ipk, inverseVcF, CAPACITANCE)

        # Update membrane potential
        u_new[i, j, k] -= dt * (ikr + iks + ik1 + ito + ina + ibna + ical + ibca + inak + inaca + ipca + ipk)
