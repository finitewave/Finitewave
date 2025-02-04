import numpy as np
from numba import njit, prange

from finitewave.cpuwave2D.model.luo_rudy91_2d import LuoRudy912D
from finitewave.cpuwave3D.stencil.isotropic_stencil_3d import (
    IsotropicStencil3D
)
from finitewave.cpuwave3D.stencil.asymmetric_stencil_3d import (
    AsymmetricStencil3D
)


class LuoRudy913D(LuoRudy912D):
    """
    Implements the 3D Luo-Rudy 1991 cardiac model.
    """
    def __init__(self):
        super().__init__()

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel to update the state variables and membrane
        potential.
        """
        ionic_kernel_3d(self.u_new, self.u, self.m, self.h, self.j_, self.d,
                        self.f, self.x, self.Cai_c,
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
            return IsotropicStencil3D()

        return AsymmetricStencil3D()


@njit(parallel=True)
def ionic_kernel_3d(u_new, u, m, h, j_, d, f, x, Cai_c, indexes, dt):
    """
    Computes the ionic currents and updates the state variables in the 3D
    Luo-Rudy 1991 cardiac model.

    Parameters
    ----------
    u_new : np.ndarray
        Array to store the updated membrane potential.
    u : np.ndarray
        Array of the current membrane potential values.
    m : np.ndarray
        Array for the gating variable `m`.
    h : np.ndarray
        Array for the gating variable `h`.
    j_ : np.ndarray
        Array for the gating variable `j_`.
    d : np.ndarray
        Array for the gating variable `d`.
    f : np.ndarray
        Array for the gating variable `f`.
    x : np.ndarray
        Array for the gating variable `x`.
    Cai_c : np.ndarray
        Array for the intracellular calcium concentration.
    mesh : np.ndarray
        Mesh array indicating the tissue types.
    dt : float
        Time step for the simulation.
    """
    Ko_c = 5.4
    Ki_c = 145
    Nai_c = 18
    Nao_c = 140
    Cao_c = 1.8

    R = 8.314
    T = 310  # 37 cels
    F = 96.5

    PR_NaK = 0.01833
    E_Na = (R*T/F)*np.log(Nao_c/Nai_c)

    n_j = u.shape[1]
    n_k = u.shape[2]

    for ind in prange(len(indexes)):
        ii = indexes[ind]
        i = ii//(n_j*n_k)
        j = (ii % (n_j*n_k))//n_k
        k = (ii % (n_j*n_k)) % n_k

        I_Na = 23*np.power(m[i, j, k], 3)*h[i, j, k] * \
            j_[i, j, k]*(u[i, j, k]-E_Na)

        alpha_h = 0
        beta_h = 0
        beta_J = 0
        alpha_J = 0
        if u[i, j, k] >= -40.:
            alpha_h = 0
            beta_h = 1./(0.13*(1 + np.exp((u[i, j, k] + 10.66)/-11.1)))
            beta_J = 0.3 * \
                np.exp(-2.535*1e-07*u[i, j, k]) / \
                (1 + np.exp(-0.1*(u[i, j, k]+32)))
            alpha_J = 0.
        else:
            alpha_h = 0.135*np.exp((80+u[i, j, k])/-6.8)
            beta_h = 3.56*np.exp(0.079*u[i, j, k]) + \
                3.1*100000*np.exp(0.35*u[i, j, k])
            beta_J = 0.1212 * \
                np.exp(-0.01052*u[i, j, k]) / \
                (1+np.exp(-0.1378*(u[i, j, k]+40.14)))
            alpha_J = (-1.2714*100000*np.exp(0.2444*u[i, j, k])-3.474*1e-05*np.exp(-0.04391*u[i, j, k]))*(
                u[i, j, k]+37.78)/(1+np.exp(0.311*(u[i, j, k]+79.23)))

        alpha_m = 0.32*(u[i, j, k]+47.13)/(1-np.exp(-0.1*(u[i, j, k]+47.13)))
        beta_m = 0.08*np.exp(-u[i, j, k]/11)

        tau_m = 1./(alpha_m+beta_m)
        inf_m = alpha_m/(alpha_m + beta_m)
        m[i, j, k] += dt*(inf_m - m[i, j, k])/tau_m

        tau_h = 1./(alpha_h+beta_h)
        inf_h = alpha_h/(alpha_h + beta_h)
        h[i, j, k] += dt*(inf_h - h[i, j, k])/tau_h

        tau_J = 1./(alpha_J+beta_J)
        inf_J = alpha_J/(alpha_J + beta_J)
        j_[i, j, k] += dt*(inf_J - j_[i, j, k])/tau_J

        # Slow inward current:
        E_Si = 7.7-13.0287*np.log(Cai_c[i, j, k])
        I_Si = 0.045*d[i, j, k]*f[i, j, k]*(u[i, j, k]-E_Si)
        alpha_d = 0.095 * \
            np.exp(-0.01*(u[i, j, k]-5))/(1+np.exp(-0.072*(u[i, j, k]-5)))
        beta_d = 0.07*np.exp(-0.017*(u[i, j, k]+44)) / \
            (1+np.exp(0.05*(u[i, j, k]+44)))
        alpha_f = 0.012 * \
            np.exp(-0.008*(u[i, j, k]+28))/(1+np.exp(0.15*(u[i, j, k]+28)))
        beta_f = 0.0065 * \
            np.exp(-0.02*(u[i, j, k]+30))/(1+np.exp(-0.2*(u[i, j, k]+30)))
        Cai_c[i, j, k] += dt*(-0.0001*I_Si+0.07*(0.0001-Cai_c[i, j, k]))

        tau_d = 1./(alpha_d+beta_d)
        inf_d = alpha_d/(alpha_d + beta_d)
        d[i, j, k] += dt*(inf_d - d[i, j, k])/tau_d

        tau_f = 1./(alpha_f+beta_f)
        inf_f = alpha_f/(alpha_f + beta_f)
        f[i, j, k] += dt*(inf_f - f[i, j, k])/tau_f

        # Time-dependent potassium current
        E_K = (R*T/F)*np.log((Ko_c + PR_NaK*Nao_c)/(Ki_c + PR_NaK*Nai_c))

        G_K = 0.705*np.sqrt(Ko_c/5.4)

        Xi = 0
        if u[i, j, k] > -100:
            Xi = 2.837*(np.exp(0.04*(u[i, j, k]+77))-1) / \
                ((u[i, j, k]+77)*np.exp(0.04*(u[i, j, k]+35)))
        else:
            Xi = 1

        I_K = G_K*x[i, j, k]*Xi*(u[i, j, k]-E_K)

        alpha_x = 0.0005 * \
            np.exp(0.083*(u[i, j, k]+50))/(1+np.exp(0.057*(u[i, j, k]+50)))
        beta_x = 0.0013 * \
            np.exp(-0.06*(u[i, j, k]+20))/(1+np.exp(-0.04*(u[i, j, k]+20)))

        tau_x = 1./(alpha_x+beta_x)
        inf_x = alpha_x/(alpha_x + beta_x)
        x[i, j, k] += dt*(inf_x - x[i, j, k])/tau_x

        # Time-independent potassium current:
        E_K1 = (R*T/F)*np.log(Ko_c/Ki_c)

        alpha_K1 = 1.02/(1+np.exp(0.2385*(u[i, j, k]-E_K1-59.215)))
        beta_K1 = (0.49124*np.exp(0.08032*(u[i, j, k]-E_K1+5.476))+np.exp(
            0.06175*(u[i, j, k]-E_K1-594.31)))/(1+np.exp(-0.5143*(u[i, j, k]-E_K1+4.753)))

        K_1x = alpha_K1/(alpha_K1+beta_K1)

        G_K1 = 0.6047*np.sqrt(Ko_c/5.4)
        I_K1 = G_K1*K_1x*(u[i, j, k]-E_K1)

        # Plateau potassium current:
        E_Kp = E_K1
        K_p = 1./(1+np.exp((7.488-u[i, j, k])/5.98))
        I_Kp = 0.0183*K_p*(u[i, j, k]-E_Kp)

        # Background current:
        I_b = 0.03921*(u[i, j, k]+59.87)

        # Total time-independent potassium current:
        I_K1_T = I_K1 + I_Kp + I_b

        u_new[i, j, k] += dt * (I_Na + I_Si + I_K1_T + I_K)
