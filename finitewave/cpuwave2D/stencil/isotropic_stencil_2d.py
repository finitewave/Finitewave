import numpy as np
from numba import njit, prange

from finitewave.core.stencil.stencil import Stencil


class IsotropicStencil2D(Stencil):
    """
    This class computes the weights for diffusion on a 2D using an isotropic
    stencil. The stencil includes 5 points: the central point and the
    four neighbors.

    The method assumes weights being used in the following order:

    - ``w[i, j, 0] : (i-1, j)``,
    - ``w[i, j, 1] : (i, j-1)``,
    - ``w[i, j, 2] : (i, j)``,
    - ``w[i, j, 3] : (i, j+1)``,
    - ``w[i, j, 4] : (i-1, j)``.

    Notes
    -----
    The method can handle heterogeneity in the diffusion coefficients given
    by the ``conductivity`` parameter.
    """

    def __init__(self):
        super().__init__()

    def select_diffusion_kernel(self):
        """
        Returns the diffusion kernel function for isotropic diffusion in 2D.

        Returns
        -------
        function
            The diffusion kernel function for isotropic diffusion in 2D.
        """
        return diffusion_kernel_2d_iso

    def compute_weights(self, model, cardiac_tissue):
        """
        Computes the weights for isotropic diffusion in 2D.

        Parameters
        ----------
        model : CardiacModel2D
            A model object containing the simulation parameters.
        cardiac_tissue : CardiacTissue2D
            A 2D cardiac tissue object.

        Returns
        -------
        numpy.ndarray
            The weights for isotropic diffusion in 2D.
        """
        mesh = cardiac_tissue.mesh.copy()
        mesh[mesh != 1] = 0
        # make sure the conductivity is a array
        conductivity = cardiac_tissue.conductivity
        conductivity = conductivity * np.ones_like(mesh, dtype=model.npfloat)
        d_xx, d_yy = self.compute_half_step_diffusion(mesh, conductivity)

        weights = np.zeros((*mesh.shape, 5), dtype=model.npfloat)
        weights = compute_weights(weights, mesh, d_xx, d_yy)
        weights = weights * model.D_model * model.dt / model.dr**2
        weights[:, :, 2] += 1

        return weights

    def compute_half_step_diffusion(self, mesh, conductivity, num_axes=2):
        """
        Computes the half-step diffusion values for isotropic diffusion.

        Parameters
        ----------
        mesh : numpy.ndarray
            A 2D array representing the mesh of the tissue.
        conductivity : numpy.ndarray
            A 2D array representing the conductivity of the tissue.
        num_axes : int
            The number of axes to compute the half-step diffusion values.

        Returns
        -------
        numpy.ndarray
            The half-step diffusion values for the specified axis.
        """
        D = np.zeros((num_axes, *mesh.shape))

        for i in range(num_axes):
            D[i] = 0.5 * (conductivity + np.roll(conductivity, -1, axis=i))

        return D


@njit(parallel=True)
def diffusion_kernel_2d_iso(u_new, u, w, indexes):
    """
    Performs isotropic diffusion on a 2D grid.

    Parameters
    ----------
    u_new : numpy.ndarray
        A 2D array to store the updated potential values after diffusion.
    u : numpy.ndarray
        A 2D array representing the current potential values before diffusion.
    w : numpy.ndarray
        A 3D array of weights used in the diffusion computation.
        The shape should match (*mesh.shape, 5).
    mesh : numpy.ndarray
        A 2D array representing the mesh of the tissue.

    Returns
    -------
    numpy.ndarray
        The updated potential values after diffusion.
    """
    n_i = u.shape[0]
    n_j = u.shape[1]
    for ind in prange(len(indexes)):
        ii = indexes[ind]
        i = int(ii / n_j)
        j = ii % n_j

        u_new[i, j] = (u[i-1, j] * w[i, j, 0] +
                       u[i, j-1] * w[i, j, 1] +
                       u[i, j] * w[i, j, 2] +
                       u[i, j+1] * w[i, j, 3] +
                       u[i+1, j] * w[i, j, 4])

    return u_new


@njit
def compute_component(d, m0, m1):
    """
    Computes the component for isotropic diffusion in 2D.

    .. code-block:: text

        m0 -- d -- x ------ m1

    Parameters
    ----------
    d : float
        The diffusion coefficient.
    m0 : int
        The value of the mesh point adjacent to the central point.
    m1 : int
        The value of the mesh point opposite to the ``m0`` point.

    Returns
    -------
    float
        The computed component for isotropic diffusion in 2D.
    """
    return d * m0 * (m0 + (m1 == 0))


@njit(parallel=True)
def compute_weights(w, m, d_xx, d_yy):
    """
    Computes the weights for isotropic diffusion in 2D.

    Parameters
    ----------
    w : numpy.ndarray
        A 3D array to store the computed weights.
    m : numpy.ndarray
        A 2D array representing the mesh of the tissue.
    d_xx : numpy.ndarray
        A 2D array representing the half-step diffusion values in the x-axis.
    d_yy : numpy.ndarray
        A 2D array representing the half-step diffusion values in the y-axis.

    Returns
    -------
    numpy.ndarray
        The computed weights for isotropic diffusion in 2D.
    """
    n_i = m.shape[0]
    n_j = m.shape[1]

    for ii in prange(n_i * n_j):

        i = int(ii / n_j)
        j = ii % n_j

        if m[i, j] != 1:
            continue

        # (i-1, j)
        w[i, j, 0] = compute_component(d_xx[i-1, j], m[i-1, j], m[i+1, j])
        # (i, j-1)
        w[i, j, 1] = compute_component(d_yy[i, j-1], m[i, j-1], m[i, j+1])
        # (i, j+1)
        w[i, j, 3] = compute_component(d_yy[i, j], m[i, j+1], m[i, j-1])
        # (i+1, j)
        w[i, j, 4] = compute_component(d_xx[i, j], m[i+1, j], m[i-1, j])
        # (i, j)
        w[i, j, 2] = - (w[i, j, 0] + w[i, j, 1] + w[i, j, 3] + w[i, j, 4])

    return w
