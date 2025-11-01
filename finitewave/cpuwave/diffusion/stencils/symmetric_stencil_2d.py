import numpy as np
from .asymmetric_stencil import AsymmetricStencil


class SymmetricStencil2D(AsymmetricStencil):
    """
    This class computes indexes and weights for the symmetric stencil.

    References
    ----------
    Bram van Es, Barry Koren, Hugo J. de Blank,
    Finite-difference schemes for anisotropic diffusion,
    Journal of Computational Physics,
    Volume 272, 2014, Pages 526-549, ISSN 0021-9991,
    https://doi.org/10.1016/j.jcp.2014.04.046.


    Notes
    -----
    The stencil is defined for 2D diffusion tensors only.
    """
    def __init__(self):
        super().__init__()

    def compute_weights(self, mesh, diffusion, indexes):
        """
        Computes the weights for the symmetric stencil.

        Parameters
        ----------
        mesh : np.ndarray
            The mesh of the simulation.
        diffusion : np.ndarray
            The diffusion tensor as a (*mesh.shape, ndim, ndim).
        indexes : np.ndarray
            The indexes of the non-empty cells in the mesh.

        Returns
        -------
        tuple
            A tuple containing:
            - rows: np.ndarray
                The row indexes for the sparse matrix.
            - cols: np.ndarray
                The column indexes for the sparse matrix.
            - weights: np.ndarray
                The weights for the sparse matrix.
        """

        if mesh.ndim != 2:
            raise ValueError("SymmetricStencil2D only supports 2D meshes.")

        return super().compute_weights(mesh, diffusion, indexes)

    def compute_flow_weights(self, mesh, diffusion, ijk, major_axis,
                             minor_axis):
        """
        Calculates the flow from major to center.

        .. code-block:: text
                m3 ---------- m4
                |             |
                |             |
                m ---- d ---- m0
                |             |
                |             |
                m1 ---------- m2
        """
        # parent class uses list for minor_axis, here we have only one
        minor_axis = minor_axis[0]

        ijk_list, mask_list = self.valid_neighbors(mesh, ijk, major_axis,
                                                   minor_axis)
        ijk, ijk_0, ijk_1, ijk_2, ijk_3, ijk_4 = ijk_list
        m, m0, m1, m2, m3, m4 = mask_list

        ijk_upper = [ijk, ijk_0, ijk_3, ijk_4]
        ijk_lower = [ijk, ijk_0, ijk_1, ijk_2]

        d_major = diffusion[..., major_axis, major_axis]
        d_minor = diffusion[..., major_axis, minor_axis]

        d_upper = [self.average_diffusion(d_major, ijk_upper, [m, m0, m3, m4]),
                   self.average_diffusion(d_minor, ijk_upper, [m, m0, m3, m4])]
        d_lower = [self.average_diffusion(d_major, ijk_lower, [m, m0, m1, m2]),
                   self.average_diffusion(d_minor, ijk_lower, [m, m0, m1, m2])]
        w_list = self.flow_weights(d_upper, d_lower, m, m0, m1, m2, m3, m4)
        return self.nonzero_weights(mesh, ijk, ijk_0, ijk_list, w_list)

    def average_diffusion(self, diffusion, ijk_list, m_list):
        """
        Averages the diffusion tensor over valid neighbors.

        Parameters
        ----------
        diffusion : np.ndarray
            The diffusion tensor as a (*mesh.shape, ndim, ndim).
        ijk_list : list of np.ndarray
            The list of ijk indexes for neighbors.
        m_list : list of np.ndarray
            The list of masks for the neighbors.

        Returns
        -------
        np.ndarray
            The averaged diffusion tensor.
        """
        m_sum = np.sum(m_list, axis=0)
        d_avg = np.zeros(m_sum.shape, dtype=diffusion.dtype)
        for m_i, ijk_i in zip(m_list, ijk_list):
            d_avg[m_i == 1] += ((1 / m_sum[m_i == 1]) *
                                diffusion[tuple(ijk_i[:, m_i == 1])])
        return d_avg

    def valid_neighbors(self, mesh, ijk, major_axis, minor_axis):
        """
        Selects valid neighbors for the major and minor components.

        .. code-block:: text
            m3 ----- m4
            |        |
            | lower  |
            |        |
            m ------ m0
            |        |
            | upper  |
            |        |
            m1 ----- m2

        Parameters
        ----------
        mesh : np.ndarray
            The mesh of the simulation.
        ijk : np.ndarray
            The indexes of the non-empty cells in the mesh.
        major_axis : int
            The major axis index.
        minor_axis : int
            The minor axis index.

        Returns
        -------
        tuple
            A tuple containing:
            - ijk_list: list of np.ndarray
                The list of ijk indexes for neighbors.
            - mask_list: list of np.ndarray
                The list of masks for the neighbors.
        """
        major_shift = 1
        minor_lower = - 1
        minor_upper = 1

        ijk_0 = self.build_neighbor(ijk, major_shift, major_axis)
        ijk_1 = self.build_neighbor(ijk, minor_lower, minor_axis)
        ijk_2 = self.build_neighbor(ijk_0, minor_lower, minor_axis)
        ijk_3 = self.build_neighbor(ijk, minor_upper, minor_axis)
        ijk_4 = self.build_neighbor(ijk_0, minor_upper, minor_axis)

        ijk_list = [ijk, ijk_0, ijk_1, ijk_2, ijk_3, ijk_4]
        valids = [self.is_valid_neighbor(ijk, mesh) for ijk in ijk_list]
        return ijk_list, valids

    def flow_weights(self, d_upper, d_lower, m, m0, m1, m2, m3, m4):
        """
        Calculates the minor component of the flow.

        q = 0.5 * (q_upper + q_lower) = 0.5 * (d_upper_xx * dx_upper +
                                               d_lower_xx * dx_lower)

        dx_upper = 0.5 * (u3 + u4 - (u + u0))
        dx_lower = 0.5 * (u + u0 - (u1 + u2))

        .. code-block:: text
            m3 ----- m4
            |        |
            |        |
            m ------ m0
            |        |
            |        |
            m1 ----- m2
        """
        # m, m0, m3, m4
        mx = (-0.5, 0.5, -0.5, 0.5)
        my = (-0.5, -0.5, 0.5, 0.5)
        w, w0, w3, w4 = self.flow_component(d_upper, m, m0, m3, m4, mx, my)
        # m, m0, m1, m2
        mx = (-0.5, 0.5, -0.5, 0.5)
        my = (0.5, 0.5, -0.5, -0.5)
        w_, w0_, w1, w2 = self.flow_component(d_lower, m, m0, m1, m2, mx, my)

        w = 0.5 * (w + w_)
        w0 = 0.5 * (w0 + w0_)
        w1 = 0.5 * w1
        w2 = 0.5 * w2
        w3 = 0.5 * w3
        w4 = 0.5 * w4
        return w, w0, w1, w2, w3, w4

    def flow_component(self, diffusion, m0, m1, m2, m3, mx, my):
        """
        q = d @ [dx, dy]
        q @ n = 0, with n the normal vector at the boundary

        qx = dxx * dx + dxy * dy
        dx = 0.5 * (u1 + u3) - 0.5 * (u0 + u2)
        dy = 0.5 * (u2 + u3) - 0.5 * (u0 + u1)

        dx = mx0 * u0 + mx1 * u1 + mx2 * u2 + mx3 * u3
        dy = my0 * u0 + my1 * u1 + my2 * u2 + my3 * u3

        .. code-block:: text
            m2 ----- m3
            |        |
            |        |
            |        |
            m0 ----- m1
        """

        dxx, dxy = diffusion

        w0 = np.zeros_like(m0, dtype=dxx.dtype)
        w1 = np.zeros_like(m1, dtype=dxy.dtype)
        w2 = np.zeros_like(m2, dtype=dxx.dtype)
        w3 = np.zeros_like(m3, dtype=dxx.dtype)

        mx0, mx1, mx2, mx3 = mx
        my0, my1, my2, my3 = my

        # m0 = m1 = m2 = m3 = 1
        mask = (m0 == 1) & (m1 == 1) & (m2 == 1) & (m3 == 1)
        w0[mask] = (dxx * mx0 + dxy * my0)[mask]
        w1[mask] = (dxx * mx1 + dxy * my1)[mask]
        w2[mask] = (dxx * mx2 + dxy * my2)[mask]
        w3[mask] = (dxx * mx3 + dxy * my3)[mask]

        # m2 = 0 or m3 = 0
        # qx = dxx * (m1 - m0)
        mask = ((m2 == 0) | (m3 == 0)) & (m1 != 0) & (m0 != 0)
        w0[mask] = - dxx[mask].copy()
        w1[mask] = dxx[mask].copy()

        return w0, w1, w2, w3
