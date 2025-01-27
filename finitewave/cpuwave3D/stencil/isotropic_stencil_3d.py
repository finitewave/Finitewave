import numpy as np
from numba import njit, prange

from finitewave.cpuwave2D.stencil.isotropic_stencil_2d import (
    compute_component,
    IsotropicStencil2D
)


class IsotropicStencil3D(IsotropicStencil2D):
    """
    This class computes the weights for diffusion on a 3D using an isotropic
    stencil. The stencil includes 7 points: the central point and the six
    neighbors.

    The method assumes weights being used in the following order:
        ``w[i, j, k, 0] : (i-1, j, k)``,
        ``w[i, j, k, 1] : (i, j-1, k)``,
        ``w[i, j, k, 2] : (i, j, k-1)``,
        ``w[i, j, k, 3] : (i, j, k)``,
        ``w[i, j, k, 4] : (i, j, k+1)``,
        ``w[i, j, k, 5] : (i, j+1, k)``,
        ``w[i, j, k, 6] : (i+1, j, k)``.

    Notes
    -----
    The method can handle heterogeneity in the diffusion coefficients given
    by the ``conductivity`` parameter.
    """

    def __init__(self):
        super().__init__()

    def select_diffusion_kernel(self):
        """
        Returns the diffusion kernel function for isotropic diffusion in 3D.

        Returns
        -------
        function
            The diffusion kernel function for isotropic diffusion in 3D.
        """
        return diffusion_kernel_3d_iso

    def compute_weights(self, model, cardiac_tissue):
        """
        Computes the weights for isotropic diffusion in 3D.

        Parameters
        ----------
        model : CardiacModel3D
            A model object containing the simulation parameters.
        cardiac_tissue : CardiacTissue3D
            A 3D cardiac tissue object.

        Returns
        -------
        numpy.ndarray
            The weights for isotropic diffusion in 3D.
        """
        mesh = cardiac_tissue.mesh.copy()
        mesh[mesh != 1] = 0

        conductivity = cardiac_tissue.conductivity
        conductivity = conductivity * np.ones_like(mesh, dtype=model.npfloat)

        d_xx, d_yy, d_zz = self.compute_half_step_diffusion(mesh, conductivity,
                                                            num_axes=3)

        weights = np.zeros((*mesh.shape, 7), dtype=model.npfloat)
        weights = compute_weights(weights, mesh, d_xx, d_yy, d_zz)
        weights = weights * model.D_model * model.dt / model.dr**2
        weights[:, :, :, 3] += 1
        return weights


@njit(parallel=True)
def diffusion_kernel_3d_iso(u_new, u, w, indexes):
    """
    Performs isotropic diffusion on a 3D grid.

    Parameters
    ----------
    u_new : numpy.ndarray
        A 3D array to store the updated potential values after diffusion.
    u : numpy.ndarray
        A 3D array representing the current potential values before diffusion.
    w : numpy.ndarray
        A 4D array of weights used in the diffusion computation.
        The shape should match (*mesh.shape, 7).
    indexes : numpy.ndarray
        A 1D array of indices where the diffusion should be computed.
    """
    n_j = u.shape[1]
    n_k = u.shape[2]

    for ind in prange(len(indexes)):
        ii = indexes[ind]
        i = ii//(n_j*n_k)
        j = (ii % (n_j*n_k))//n_k
        k = (ii % (n_j*n_k)) % n_k

        u_new[i, j, k] = (u[i-1, j, k] * w[i, j, k, 0] +
                          u[i, j-1, k] * w[i, j, k, 1] +
                          u[i, j, k-1] * w[i, j, k, 2] +
                          u[i, j, k] * w[i, j, k, 3] +
                          u[i, j, k+1] * w[i, j, k, 4] +
                          u[i, j+1, k] * w[i, j, k, 5] +
                          u[i+1, j, k] * w[i, j, k, 6])


@njit(parallel=True)
def compute_weights(w, m, d_xx, d_yy, d_zz):
    n_i = m.shape[0]
    n_j = m.shape[1]
    n_k = m.shape[2]

    for ii in prange(n_i * n_j * n_k):

        i = ii // (n_j * n_k)
        j = (ii % (n_j * n_k)) // n_k
        k = (ii % (n_j * n_k)) % n_k

        if m[i, j, k] != 1:
            continue

        # (i-1, j, k)
        w[i, j, k, 0] = compute_component(d_xx[i-1, j, k],
                                          m[i-1, j, k], m[i+1, j, k])
        # (i, j-1, k)
        w[i, j, k, 1] = compute_component(d_yy[i, j-1, k],
                                          m[i, j-1, k], m[i, j+1, k])
        # (i, j, k-1)
        w[i, j, k, 2] = compute_component(d_zz[i, j, k-1],
                                          m[i, j, k-1], m[i, j, k+1])
        # (i, j, k+1)
        w[i, j, k, 4] = compute_component(d_zz[i, j, k],
                                          m[i, j, k+1], m[i, j, k-1])
        # (i, j+1, k)
        w[i, j, k, 5] = compute_component(d_yy[i, j, k],
                                          m[i, j+1, k], m[i, j-1, k])
        # (i+1, j, k)
        w[i, j, k, 6] = compute_component(d_xx[i, j, k],
                                          m[i+1, j, k], m[i-1, j, k])
        # (i, j, k)
        w[i, j, k, 3] = - (w[i, j, k, 0] + w[i, j, k, 1] + w[i, j, k, 2] +
                           w[i, j, k, 4] + w[i, j, k, 5] + w[i, j, k, 6])

    return w
