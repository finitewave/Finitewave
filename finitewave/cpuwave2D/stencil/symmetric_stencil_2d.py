import numpy as np
from numba import njit, prange

from .asymmetric_stencil_2d import AsymmetricStencil2D


class SymmetricStencil2D(AsymmetricStencil2D):
    """
    A class to represent a 2D symmetric stencil for diffusion processes.
    The asymmetric stencil is used to handle anisotropic diffusion in the
    tissue.

    Notes
    -----
    The method assumes weights being used in the following order:

    - ``w[0] : i-1, j-1``,
    - ``w[1] : i-1, j``,
    - ``w[2] : i-1, j+1``,
    - ``w[3] : i, j-1``,
    - ``w[4] : i, j``,
    - ``w[5] : i, j+1``,
    - ``w[6] : i+1, j-1``,
    - ``w[7] : i+1, j``,
    - ``w[8] : i+1, j+1``.
    """

    def __init__(self):
        super().__init__()

    def compute_weights(self, model, cardiac_tissue):
        """
        Computes the weights for diffusion on a 2D mesh using the symmetric
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
            3D array of weights for diffusion, with the shape of
            ``(*mesh.shape, 9)``.
        """
        mesh = cardiac_tissue.mesh.copy()
        conductivity = cardiac_tissue.conductivity
        fibers = cardiac_tissue.fibers

        if fibers is None:
            message = "Fibers must be provided for anisotropic diffusion."
            raise ValueError(message)

        mesh[mesh != 1] = 0
        # fibers[np.where(mesh != 1)] = 0
        weights = np.zeros((*mesh.shape, 9))

        D = self.compute_half_step_diffusion(mesh, conductivity, fibers,
                                             self.D_al, self.D_ac)
        compute_weights(weights, mesh, D[0], D[1], D[2], D[3])
        weights *= model.D_model * model.dt / model.dr**2
        weights[:, :, 4] += 1

        return weights

    def compute_half_step_diffusion(self, mesh, conductivity, fibers, D_al,
                                    D_ac):
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
        D_al : float
            Longitudinal diffusion coefficient.
        D_ac : float
            Cross-sectional diffusion coefficient.
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

        D = np.zeros((4, *mesh.shape))
        for i in range(4):
            ind0 = i // 2
            ind1 = i % 2
            D[i] = self.compute_diffusion_components(fibers, ind0, ind1, D_al,
                                                     D_ac)
            D[i] *= conductivity
            D[i] = 0.25 * (D[i] +
                           np.roll(D[i], -1, axis=0) +
                           np.roll(D[i], -1, axis=1) +
                           np.roll(np.roll(D[i], -1, axis=0), -1, axis=1))

        return D


@njit
def compute_components(d_xx, d_xy, d_yx, d_yy, m0, m1, m2, m3, qx, qy):
    """
    .. code-block:: text
        m1 ---- m3
        |       |
        |   o   |
        |       |
        m0 ---- m2


    dx = 0.5 * (u2 + u3) - 0.5 * (u0 + u1)
    dy = 0.5 * (u1 + u3) - 0.5 * (u0 + u2)

    qx = d_xx * dx + d_xy * dy
    qy = d_yx * dx + d_yy * dy
    """
    m = m0 + m1 + m2 + m3

    if m < 3:
        return 0, 0, 0, 0

    qdx = qx * d_xx + qy * d_yx
    qdy = qx * d_xy + qy * d_yy

    w0 = - m0 / (m0 + m1) * qdx - m0 / (m0 + m2) * qdy
    w1 = - m1 / (m0 + m1) * qdx + m1 / (m1 + m3) * qdy
    w2 = m2 / (m2 + m3) * qdx - m2 / (m0 + m2) * qdy
    w3 = m3 / (m2 + m3) * qdx + m3 / (m1 + m3) * qdy
    return 0.5 * w0, 0.5 * w1, 0.5 * w2, 0.5 * w3


@njit
def compute_component_(m0, m1, m2, m3, d_xx, d_xy, d_yx, d_yy, qx, qy, ux, uy):
    m = m0 * m1 * m2 * m3
    w = (qx * (d_xx * ux + d_xy * uy) + qy * (d_yx * ux + d_yy * uy)) * m
    return 0.25 * w


@njit
def compute_weights(w, m, d_xx, d_xy, d_yx, d_yy):
    """
    Computes the weights for diffusion on a 2D mesh based on the asymmetric
    stencil.

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

        # (i-1/2, j-1/2)
        w0, w1, w2, w3 = compute_components(d_xx[i-1, j-1], d_xy[i-1, j-1],
                                            d_yx[i-1, j-1], d_yy[i-1, j-1],
                                            m[i-1, j-1], m[i-1, j], m[i, j-1],
                                            m[i, j], -1, -1)
        # (i-1, j-1)
        w[i, j, 0] += w0
        # (i-1, j)
        w[i, j, 1] += w1
        # (i, j-1)
        w[i, j, 3] += w2
        # (i, j)
        w[i, j, 4] += w3

        # (i-1/2, j+1/2)
        w0, w1, w2, w3 = compute_components(d_xx[i-1, j], d_xy[i-1, j],
                                            d_yx[i-1, j], d_yy[i-1, j],
                                            m[i-1, j], m[i-1, j+1], m[i, j],
                                            m[i, j+1], -1, 1)
        # (i-1, j)
        w[i, j, 1] += w0
        # (i-1, j+1)
        w[i, j, 2] += w1
        # (i, j)
        w[i, j, 4] += w2
        # (i, j+1)
        w[i, j, 5] += w3

        # (i+1/2, j-1/2)
        w0, w1, w2, w3 = compute_components(d_xx[i, j-1], d_xy[i, j-1],
                                            d_yx[i, j-1], d_yy[i, j-1],
                                            m[i, j-1], m[i, j], m[i+1, j-1],
                                            m[i+1, j], 1, -1)
        # (i, j-1)
        w[i, j, 3] += w0
        # (i, j)
        w[i, j, 4] += w1
        # (i+1, j-1)
        w[i, j, 6] += w2
        # (i+1, j)
        w[i, j, 7] += w3

        # (i+1/2, j+1/2)
        w0, w1, w2, w3 = compute_components(d_xx[i, j], d_xy[i, j],
                                            d_yx[i, j], d_yy[i, j],
                                            m[i, j], m[i, j+1], m[i+1, j],
                                            m[i+1, j+1], 1, 1)
        # (i, j)
        w[i, j, 4] += w0
        # (i, j+1)
        w[i, j, 5] += w1
        # (i+1, j)
        w[i, j, 7] += w2
        # (i+1, j+1)
        w[i, j, 8] += w3

    return w
