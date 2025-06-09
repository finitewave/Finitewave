import numpy as np
from numba import njit, prange

from finitewave.cpuwave2D.model.courtemanche_2d import (
    Courtemanche2D,
    calc_ina,
    calc_ik1,
    calc_ito,
    calc_ikr,
    calc_iks,
    calc_ikur,
    calc_ical,
    calc_inak,
    calc_inaca,
    calc_ibca,
    calc_ibna,
    calc_ipca,
    calc_irel,
    calc_iupleak,
    calc_iup,
    calc_itr,
    calc_caup,
    calc_nai,
    calc_ki,
    calc_cai,
    calc_carel,
    calc_gating_m,
    calc_gating_h,
    calc_gating_j,
    calc_equilibrum_potentials
)
from finitewave.cpuwave3D.stencil.isotropic_stencil_3d import (
    IsotropicStencil3D
)
from finitewave.cpuwave3D.stencil.asymmetric_stencil_3d import (
    AsymmetricStencil3D
)


class Courtemanche3D(Courtemanche2D):
    """
    A class to represent the Courtemanche cardiac model in 3D.

    See Courtemanche2D for the 2D model description.
    """

    def __init__(self):
        super().__init__()

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel function to update ionic currents and state
        variables.
        """
        ionic_kernel_3d(self.u_new, self.u, self.nai, self.ki, self.cai, self.caup, self.carel, 
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
            return IsotropicStencil3D()

        return AsymmetricStencil3D()


@njit(parallel=True)
def ionic_kernel_3d(u_new, u, nai, ki, cai, caup, carel, m, h, j_, d, f, oa, oi, ua, ui, xs, xr, fca, irel, vrel, urel, wrel, indexes, dt, 
                    gna, gnab, gk1, gkr, gks, gto, gcal, gcab, gkur_coeff, F, T, R, Vc, Vj, Vup, Vrel, ibk, cao, nao, ko, caupmax, kup,
                    kmnai, kmko, kmnancx, kmcancx, ksatncx, kmcmdn, kmtrpn, kmcsqn, trpnmax, cmdnmax, csqnmax, inacamax,
                    inakmax, ipcamax, krel, iupmax, kq10):
    """
    Computes the ionic currents and updates the state variables in the 3D
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
                    
    n_j = u.shape[1]
    n_k = u.shape[2]

    for ind in prange(len(indexes)):
        ii = indexes[ind]
        i = ii//(n_j*n_k)
        j = (ii % (n_j*n_k))//n_k
        k = (ii % (n_j*n_k)) % n_k

        ena, ek, eca = calc_equilibrum_potentials(nai[i, j, k], nao, ki[i, j, k], ko, cai[i, j, k], cao, R, T, F)

        m[i, j, k] = calc_gating_m(m[i, j, k], u[i, j, k], dt)
        h[i, j, k] = calc_gating_h(h[i, j, k], u[i, j, k], dt)
        j_[i, j, k] = calc_gating_j(j_[i, j, k], u[i, j, k], dt)

        ina = calc_ina(u[i, j, k], m[i, j, k], h[i, j, k], j_[i, j, k], gna, ena)

        ik1 = calc_ik1(u[i, j, k], gk1, ek)

        ito, oa[i, j, k], oi[i, j, k] = calc_ito(u[i, j, k], dt, kq10, oa[i, j, k], oi[i, j, k], gto, ek)

        ikur, ua[i, j, k], ui[i, j, k] = calc_ikur(u[i, j, k], dt, kq10, ua[i, j, k], ui[i, j, k], ek, gkur_coeff)

        ikr, xr[i, j, k] = calc_ikr(u[i, j, k], dt, xr[i, j, k], gkr, ek)

        iks, xs[i, j, k] = calc_iks(u[i, j, k], dt, xs[i, j, k], gks, ek)

        ical, d[i, j, k], f[i, j, k], fca[i, j, k] = calc_ical(u[i, j, k], dt, d[i, j, k], f[i, j, k], cai[i, j, k], gcal, fca[i, j, k], eca)

        inak = calc_inak(inakmax, nai[i, j, k], nao, ko, kmnai, kmko, F, u[i, j, k], R, T)
        inaca = calc_inaca(inacamax, nai[i, j, k], nao, cai[i, j, k], cao, kmnancx, kmcancx, ksatncx, F, u[i, j, k], R, T)

        ibca = calc_ibca(gcab, eca, u[i, j, k])

        ibna = calc_ibna(gnab, ena, u[i, j, k])

        ipca = calc_ipca(ipcamax, cai[i, j, k])

        irel[i, j, k], urel[i, j, k], vrel[i, j, k], wrel[i, j, k] = calc_irel(dt, urel[i, j, k], vrel[i, j, k], irel[i, j, k], wrel[i, j, k], ical, inaca, krel, carel[i, j, k], cai[i, j, k], u[i, j, k], F, Vrel)
        itr = calc_itr(caup[i, j, k], carel[i, j, k])
        iup = calc_iup(iupmax, cai[i, j, k], kup)
        iupleak = calc_iupleak(caup[i, j, k], caupmax, iupmax) 

        caup[i, j, k] = calc_caup(caup[i, j, k], dt, iup, iupleak, itr, Vrel, Vup)
        nai[i, j, k] = calc_nai(nai[i, j, k], dt, inak, inaca, ibna, ina, F, Vj)

        ki[i, j, k] = calc_ki(ki[i, j, k], dt, inak, ik1, ito, ikur, ikr, iks, ibk, F, Vj)
        cai[i, j, k] = calc_cai(cai[i, j, k], dt, inaca, ipca, ical, ibca, iup, iupleak, irel[i, j, k], Vrel, Vup, trpnmax, kmtrpn, cmdnmax, kmcmdn, F, Vj)

        carel[i, j, k] = calc_carel(carel[i, j, k], dt, itr, irel[i, j, k], csqnmax, kmcsqn)

        u_new[i, j, k] -= dt * (ina + ik1 + ito + ikur + ikr + iks + ical + ipca + inak + inaca + ibna + ibca)
