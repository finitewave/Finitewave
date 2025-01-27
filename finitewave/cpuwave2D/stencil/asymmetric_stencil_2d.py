import numpy as np
from numba import njit, prange

from finitewave.core.stencil.stencil import Stencil


class AsymmetricStencil2D(Stencil):
    """
    This class computes the weights for diffusion on a 2D using an asymmetric
    stencil. The weights are calculated based on diffusion coefficients and
    fiber orientations. The stencil includes 9 points: the central point and
    8 surrounding points. The boundary conditions are Neumann with first-order
    approximation.

    Attributes
    ----------
    D_al : float
        Longitudinal diffusion coefficient.
    D_ac : float
        Cross-sectional diffusion coefficient.

    Notes
    -----
    The diffusion coefficients are general and should be adjusted according to
    the specific model. These parameters only set the ratios between
    longitudinal and cross-sectional diffusion.

    The method assumes weights being used in the following order:

    - ``w[i, j, 0] : (i-1, j-1)``,
    - ``w[i, j, 1] : (i-1, j)``,
    - ``w[i, j, 2] : (i-1, j+1)``,
    - ``w[i, j, 3] : (i, j-1)``,
    - ``w[i, j, 4] : (i, j)``,
    - ``w[i, j, 5] : (i, j+1)``,
    - ``w[i, j, 6] : (i+1, j-1)``,
    - ``w[i, j, 7] : (i+1, j)``,
    - ``w[i, j, 8] : (i+1, j+1)``.
    """

    def __init__(self):
        super().__init__()
        self.D_al = 1
        self.D_ac = 1/9

    def compute_weights(self, model, cardiac_tissue):
        """
        Computes the weights for diffusion on a 2D mesh using an asymmetric
        stencil.

        Parameters
        ----------
        model : CardiacModel2D
            A model object containing the simulation parameters.
        cardiac_tissue : CardiacTissue2D
            A 2D cardiac tissue object.

        Returns
        -------
        np.ndarray
            Array of weights for diffusion with the shape of (*mesh.shape, 9).
        """
        # convert fibrotic areas to non-tissue
        mesh = cardiac_tissue.mesh.copy()
        mesh[mesh != 1] = 0
        conductivity = cardiac_tissue.conductivity
        conductivity = conductivity * np.ones_like(mesh, dtype=model.npfloat)

        fibers = cardiac_tissue.fibers

        if fibers is None:
            message = "Fibers must be provided for anisotropic diffusion."
            raise ValueError(message)

        weights = np.zeros((*mesh.shape, 9))
        d_xx, d_xy = self.compute_half_step_diffusion(mesh, conductivity,
                                                      fibers, 0)
        d_yx, d_yy = self.compute_half_step_diffusion(mesh, conductivity,
                                                      fibers, 1)
        weights = compute_weights(weights, mesh, d_xx, d_xy, d_yx, d_yy)
        weights = weights * model.D_model * model.dt / model.dr**2
        weights[:, :, 4] += 1

        return weights

    def select_diffusion_kernel(self):
        """
        Returns the diffusion kernel function for anisotropic diffusion in 2D.

        Returns
        -------
        function
            The diffusion kernel function for anisotropic diffusion in 2D.
        """
        return diffusion_kernel_2d_aniso

    def compute_half_step_diffusion(self, mesh, conductivity, fibers, axis,
                                    num_axes=2):
        """
        Computes the diffusion components for half-steps based on fiber
        orientations.

        Parameters
        ----------
        mesh : np.ndarray
            Array representing the mesh grid of the tissue.
        conductivity : np.ndarray
            Array representing the conductivity of the tissue.
        fibers : np.ndarray
            Array representing fiber orientations with shape
            ``(2, *mesh.shape)``.
        axis : int
            Axis index (0 for x, 1 for y).
        num_axes : int
            Number of axes.

        Returns
        -------
        np.ndarray
            Array of diffusion components for half-steps along the specified
            axis.

        Notes
        -----
        The index ``i`` in the returned array corresponds to ``i+1/2`` and
        ``i-1`` corresponds to ``i-1/2``.
        """
        D = np.zeros((num_axes, *mesh.shape))
        for i in range(num_axes):
            D[i] = self.compute_diffusion_components(fibers, axis, i,
                                                     self.D_al, self.D_ac)
            D[i] = 0.5 * (D[i] * conductivity +
                          np.roll(D[i], -1, axis=axis) *
                          np.roll(conductivity, -1, axis=axis))

        return D

    def compute_diffusion_components(self, fibers, ind0, ind1, D_al, D_ac):
        """
        Computes the diffusion components based on fiber orientations.

        Parameters
        ----------
        fibers : np.ndarray
            Array representing fiber orientations.
        ind0 : int
            First axis index (0 for x, 1 for y).
        ind1 : int
            Second axis index (0 for x, 1 for y).
        D_al : float
            Longitudinal diffusion coefficient.
        D_ac : float
            Cross-sectional diffusion coefficient.

        Returns
        -------
        np.ndarray
            Array of diffusion components based on fiber orientations
        """
        return (D_ac * (ind0 == ind1) +
                (D_al - D_ac) * fibers[..., ind0] * fibers[..., ind1])


@njit(parallel=True)
def diffusion_kernel_2d_aniso(u_new, u, w, indexes):
    """
    Performs anisotropic diffusion on a 2D grid.

    Parameters
    ----------
    u_new : np.ndarray
        Array to store the updated potential values after diffusion.
    u : np.ndarray
        Array representing the current potential values before diffusion.
    w : np.ndarray
        Array of weights used in the diffusion computation.
    mesh : np.ndarray
        Array representing the mesh of the tissue.

    Returns
    -------
    np.ndarray
        The updated potential values after diffusion.
    """
    n_i = u.shape[0]
    n_j = u.shape[1]
    for ind in prange(len(indexes)):
        ii = indexes[ind]
        i = int(ii / n_j)
        j = ii % n_j

        u_new[i, j] = (u[i-1, j-1] * w[i, j, 0] +
                       u[i-1, j] * w[i, j, 1] +
                       u[i-1, j+1] * w[i, j, 2] +
                       u[i, j-1] * w[i, j, 3] +
                       u[i, j] * w[i, j, 4] +
                       u[i, j+1] * w[i, j, 5] +
                       u[i+1, j-1] * w[i, j, 6] +
                       u[i+1, j] * w[i, j, 7] +
                       u[i+1, j+1] * w[i, j, 8])
    return u_new


@njit
def minor_component(d, m0, m1, m2, m3, m4, m5):
    """
    Calculates the minor component for the diffusion current.

    .. code-block:: text
        m4 ----- m5
        |        |
        |        |
        |        |
        m2 - d - m3
        |        |
        |        |
        |        |
        m0 ----- m1

    Parameters
    ----------
    d : float
        Minor diffusion at half-steps.
    m0 : int
        Mesh point value at (i-1, j-1).
    m1 : int
        Mesh point value at (i-1, j).
    m2 : int
        Mesh point value at (i, j-1).
    m3 : int
        Mesh point value at (i, j).
    m4 : int
        Mesh point value at (i+1, j-1).
    m5 : int
        Mesh point value at (i+1, j).

    Returns
    -------
    tuple
        Tuple of weights for each of the 6 points.

    Notes
    -----
    The order of the points assumes m3 is the central point of the stencil.
    """
    m_higher = m2 + m3 + m4 + m5
    m_lower = m0 + m1 + m2 + m3

    if m2 == 0 or m3 == 0 or m_higher < 3 or m_lower < 3:
        return 0, 0, 0, 0, 0, 0

    w0 = - d * m0 / m_lower
    w1 = - d * m1 / m_lower
    w2 = d * (m2 / m_higher - m2 / m_lower)
    w3 = d * (m3 / m_higher - m3 / m_lower)
    w4 = d * m4 / m_higher
    w5 = d * m5 / m_higher

    return w0, w1, w2, w3, w4, w5


@njit
def major_component(d, m0):
    """
    Computes the major component for the difussion current.

    .. code-block:: text
        x ------ x
        |        |
        |        |
        m0 - d - m1
        |        |
        |        |
        x ------ x

    Parameters
    ----------
    d : np.ndarray
        Major diffusion at half-steps.
    m0 : np.ndarray
        Mesh point adjacent to the central point.

    Returns
    -------
    np.ndarray
        Major component for the diffusion.
    """
    return d * m0


@njit(parallel=True)
def compute_weights(w, m, d_xx, d_xy, d_yx, d_yy):
    """
    Computes the weights for diffusion on a 2D mesh based on the asymmetric
    stencil.

    .. code-block:: text
        w2 --------------- w5 ---------------- w8
        |                  |                   |
        |               d_yy_1                 |
        |               d_yx_1                 |
        |                  |                   |
        |                  |                   |
        w1 ---- d_xx_0 --- w4 ---- d_xx_1 ---- w7
        |       d_xy_0     |       d_xy_1      |
        |                  |                   |
        |               d_yy_0                 |
        |               d_yx_0                 |
        |                  |                   |
        w0 --------------- w3 ---------------- w6

    Parameters
    ----------
    w : np.ndarray
        3D array to store the weights for diffusion. Shape is (*mesh.shape, 9).
    m : np.ndarray
        2D array representing the mesh grid of the tissue. Non-tissue areas
        are set to 0.
    d_xx : np.ndarray
        Diffusion x component for x direction.
    d_xy : np.ndarray
        Diffusion y component for x direction.
    d_yx : np.ndarray
        Diffusion x component for y direction.
    d_yy : np.ndarray
        Diffusion y component for y direction.

    Returns
    -------
    np.ndarray
        3D array of weights for diffusion, with the shape of (*mesh.shape, 9).
    """
    n_i = m.shape[0]
    n_j = m.shape[1]
    for ii in prange(n_i * n_j):
        i = int(ii / n_j)
        j = ii % n_j
        if m[i, j] != 1:
            continue

        # q (i-1/2, j)
        qx0_major = major_component(d_xx[i-1, j], m[i-1, j])
        # (i-1, j)
        w[i, j, 1] += qx0_major
        # (i, j)
        w[i, j, 4] -= qx0_major

        qx0_minor = minor_component(d_xy[i-1, j],
                                    m[i-1, j-1], m[i, j-1],
                                    m[i-1, j], m[i, j],
                                    m[i-1, j+1], m[i, j+1])
        # (i-1, j-1)
        w[i, j, 0] -= qx0_minor[0]
        # (i, j-1)
        w[i, j, 3] -= qx0_minor[1]
        # (i-1, j)
        w[i, j, 1] -= qx0_minor[2]
        # (i, j)
        w[i, j, 4] -= qx0_minor[3]
        # (i-1, j+1)
        w[i, j, 2] -= qx0_minor[4]
        # (i, j+1)
        w[i, j, 5] -= qx0_minor[5]

        # q (i, j-1/2)
        qy0_major = major_component(d_yy[i, j-1], m[i, j-1])
        # (i, j-1)
        w[i, j, 3] += qy0_major
        # (i, j)
        w[i, j, 4] -= qy0_major

        qy0_minor = minor_component(d_yx[i, j-1], m[i-1, j-1], m[i-1, j],
                                    m[i, j-1], m[i, j], m[i+1, j-1], m[i+1, j])

        # (i-1, j-1)
        w[i, j, 0] -= qy0_minor[0]
        # (i-1, j)
        w[i, j, 1] -= qy0_minor[1]
        # (i, j-1)
        w[i, j, 3] -= qy0_minor[2]
        # (i, j)
        w[i, j, 4] -= qy0_minor[3]
        # (i+1, j-1)
        w[i, j, 6] -= qy0_minor[4]
        # (i+1, j)
        w[i, j, 7] -= qy0_minor[5]

        # q (i, j+1/2)
        qy1_major = major_component(d_yy[i, j], m[i, j+1])
        # (i, j+1)
        w[i, j, 5] += qy1_major
        # (i, j)
        w[i, j, 4] -= qy1_major

        qy1_minor = minor_component(d_yx[i, j], m[i-1, j+1], m[i-1, j],
                                    m[i, j+1], m[i, j], m[i+1, j+1], m[i+1, j])

        # (i-1, j+1)
        w[i, j, 2] += qy1_minor[0]
        # (i-1, j)
        w[i, j, 1] += qy1_minor[1]
        # (i, j+1)
        w[i, j, 5] += qy1_minor[2]
        # (i, j)
        w[i, j, 4] += qy1_minor[3]
        # (i+1, j+1)
        w[i, j, 8] += qy1_minor[4]
        # (i+1, j)
        w[i, j, 7] += qy1_minor[5]

        # q (i+1/2, j)
        qx1_major = major_component(d_xx[i, j], m[i+1, j])
        # (i+1, j)
        w[i, j, 7] += qx1_major
        # (i, j)
        w[i, j, 4] -= qx1_major

        qx1_minor = minor_component(d_xy[i, j], m[i+1, j-1], m[i, j-1],
                                    m[i+1, j], m[i, j], m[i+1, j+1], m[i, j+1])
        # (i+1, j-1)
        w[i, j, 6] += qx1_minor[0]
        # (i, j-1)
        w[i, j, 3] += qx1_minor[1]
        # (i+1, j)
        w[i, j, 7] += qx1_minor[2]
        # (i, j)
        w[i, j, 4] += qx1_minor[3]
        # (i+1, j+1)
        w[i, j, 8] += qx1_minor[4]
        # (i, j+1)
        w[i, j, 5] += qx1_minor[5]

    return w
