import numpy as np


class SymmetricStencil:
    """
    This class computes indexes and weights for the symmetric stencil.

    Notes
    -----
    The symmetric stencil reduces to the isotropic stencil with first-order
    boundary conditions if the ``fibers = None`` or ``fibers`` are aligned with
    the grid and ``D_al = D_ac``.

    Rules for handling boundaries are:
    - If a major (directly adjacent) neighbor is invalid
        (out of bounds or in an empty cell), flow from this neighbor is zero.
    - If more than one minor neighbor from upper or lower side is invalid,
        flow from these neighbors is determined only by the major component.
    - If a minor neighbor from upper or lower side is invalid, minor component
        is calculated using the remaining valid minor neighbors.

    Diffusion components are calculated in the middle of two nodes, therefore
    the diffusion coefficient is averaged between the corresponding nodes.

    References
    ----------
    Bram van Es, Barry Koren, Hugo J. de Blank,
    Finite-difference schemes for anisotropic diffusion,
    Journal of Computational Physics,
    Volume 272, 2014, Pages 526-549, ISSN 0021-9991,
    https://doi.org/10.1016/j.jcp.2014.04.046.
    """
    def __init__(self):
        pass

    def compute_weights(self, mesh, diffusion, indexes):
        """
        Computes the weights for the asymmetric stencil.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        diffusion : numpy.ndarray
            The diffusion tensor as a (*mesh.shape, ndim, ndim).
        indexes : numpy.ndarray
            The indexes of the non-empty cells in the mesh.
        """
        rows, cols, weights = [], [], []

        print(diffusion[0, 0, :, :])
        for i in range(mesh.ndim):
            axis = np.roll(np.arange(mesh.ndim), i)
            major_axis = axis[0]
            minor_axes = axis[1:]
            for minor_axis in minor_axes:
                D = diffusion[..., [major_axis, minor_axis]][..., [major_axis, minor_axis], :].copy()

                for major_neighbor in [-1, 1]:
                    res = self.compute_flow_weights(D, major_axis, minor_axis,
                                                    major_neighbor, mesh, indexes)
                    rows += res[0]
                    cols += res[1]
                    weights += res[2]

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights)
        return rows, cols, weights

    def compute_flow_weights(self, D, major_axis, minor_axis, major_neighbor,
                             mesh, indexes):
        """
        Calculates the flow from major to center.

        .. code-block:: text
             minor_3 ----------- minor_4
                |                   |
                |                   |
                |                   |
              major ---- d ---- center
                |                   |
                |                   |
                |                   |
             minor_1 ----------- minor_2
        """
        ijk_list, mask_list = self.valid_neighbors(mesh, indexes, major_axis,
                                                   minor_axis, major_neighbor)
        ijk, ijk_0, ijk_1, ijk_2, ijk_3, ijk_4 = ijk_list

        print(major_axis, minor_axis, major_neighbor)

        m, m0, m1, m2, m3, m4 = mask_list
        d_upper = np.zeros((ijk.shape[1], 2, 2), dtype=D.dtype)

        m_upper = m + m0 + m3 + m4
        for m_i, ijk_i in zip([m, m0, m3, m4], [ijk, ijk_0, ijk_3, ijk_4]):
            d_upper[m_i == 1] += ((1 / m_upper[m_i == 1, None, None]) *
                                  (D[tuple(ijk_i[:, m_i == 1])]))

        d_lower = np.zeros((ijk.shape[1], 2, 2), dtype=D.dtype)
        m_lower = m + m0 + m1 + m2
        for m_i, ijk_i in zip([m, m0, m1, m2], [ijk, ijk_0, ijk_1, ijk_2]):
            d_lower[m_i == 1] += (1 / m_lower[m_i == 1, None, None] *
                                  (D[tuple(ijk_i[:, m_i == 1])]))

        weights = self.flow_weights(d_upper, d_lower, *mask_list)

        rows = [np.ravel_multi_index(ijk[:, w != 0], mesh.shape)
                for w in weights]
        cols = [np.ravel_multi_index(ijk[:, w != 0], mesh.shape)
                for ijk, w in zip(ijk_list, weights)]
        weights = [w[w != 0] for w in weights]

        return rows, cols, weights

    def valid_neighbors(self, mesh, indexes, major_axis, minor_axis,
                        major_shift):
        """
        Selects valid neighbors for the major and minor components.

        .. code-block:: text
             minor_3 ------------ minor_4
                |                   |
                |                   |
                |                   |
              center ------ d ---- major
                |                   |
                |                   |
                |                   |
             minor_1 ------------ minor_2
        """
        minor_lower = major_shift
        minor_upper = -major_shift

        ijk_center = np.array(np.unravel_index(indexes, mesh.shape))
        ijk_major = self.build_neighbor(ijk_center, major_shift, major_axis)
        ijk_minor_1 = self.build_neighbor(ijk_center, minor_lower, minor_axis)
        ijk_minor_2 = self.build_neighbor(ijk_major, minor_lower, minor_axis)
        ijk_minor_3 = self.build_neighbor(ijk_center, minor_upper, minor_axis)
        ijk_minor_4 = self.build_neighbor(ijk_major, minor_upper, minor_axis)

        ijk_list = [ijk_center, ijk_major, ijk_minor_1, ijk_minor_2,
                    ijk_minor_3, ijk_minor_4]

        valids = [self.is_valid_neighbor(ijk, mesh) for ijk in ijk_list]
        return ijk_list, valids

    def build_neighbor(self, ijk, shift, axis):
        """
        Builds the neighbor ijk coordinates by shifting along a given axis.

        Parameters
        ----------
        ijk : numpy.ndarray
            The ijk coordinates of the current cell.
        shift : int
            The shift to apply along the specified axis.
        axis : int
            The axis along which to apply the shift.

        Returns
        -------
        numpy.ndarray
            The ijk coordinates of the neighbor.
        """

        ijk = ijk.copy()
        ijk[axis] += shift
        return ijk

    def is_valid_neighbor(self, neighbor, mesh):
        """
        Checks if the neighbor is valid
        (i.e., within the mesh and not empty).
        """
        mask = np.ones(neighbor.shape[1], dtype=bool)

        for i in range(mesh.ndim):
            mask &= ((neighbor[i] >= 0) & (neighbor[i] < mesh.shape[i]))

        mask[mask] = mesh[tuple(neighbor[:, mask])] == 1
        return mask.astype(mesh.dtype)

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
            |        |
            m ------ m0
            |        |
            |        |
            |        |
            m1 ----- m2
        """
        w = np.zeros_like(m, dtype=d_upper.dtype)
        w0 = np.zeros_like(m0, dtype=d_upper.dtype)
        w1 = np.zeros_like(m1, dtype=d_upper.dtype)
        w2 = np.zeros_like(m2, dtype=d_upper.dtype)
        w3 = np.zeros_like(m3, dtype=d_upper.dtype)
        w4 = np.zeros_like(m4, dtype=d_upper.dtype)

        # m, m0, m3, m4
        w_, w0_, w3_, w4_ = self.flow_upper_component(d_upper, m, m0, m3, m4)
        w += 0.5 * w_
        w0 += 0.5 * w0_
        w3 += 0.5 * w3_
        w4 += 0.5 * w4_

        # m, m0, m1, m2
        w_, w0_, w1_, w2_ = self.flow_lower_component(d_lower, m, m0, m1, m2)
        w += 0.5 * w_
        w0 += 0.5 * w0_
        w1 += 0.5 * w1_
        w2 += 0.5 * w2_
        return w, w0, w1, w2, w3, w4

    def flow_upper_component(self, d, m0, m1, m2, m3):
        """
        q = d @ [dx, dy]
        q @ n = 0, with n the normal vector at the boundary

        qx = dxx * dx + dxy * dy
        dx = 0.5 * (u1 + u3) - 0.5 * (u0 + u2)
        dy = 0.5 * (u2 + u3) - 0.5 * (u0 + u1)

        .. code-block:: text
            m2 ----- m3
            |        |
            |        |
            |        |
            m0 ----- m1
        """
        w0 = np.zeros_like(m0, dtype=d.dtype)
        w1 = np.zeros_like(m1, dtype=d.dtype)
        w2 = np.zeros_like(m2, dtype=d.dtype)
        w3 = np.zeros_like(m3, dtype=d.dtype)

        dxx = d[:, 0, 0]
        dxy = d[:, 0, 1]
        dyx = d[:, 1, 0]
        dyy = d[:, 1, 1]

        dx0, dy0 = - 0.5, - 0.5
        dx1, dy1 = 0.5, - 0.5
        dx2, dy2 = - 0.5, 0.5
        dx3, dy3 = 0.5, 0.5

        coef0 = dxx * dx0 + dxy * dy0
        coef1 = dxx * dx1 + dxy * dy1
        coef2 = dxx * dx2 + dxy * dy2
        coef3 = dxx * dx3 + dxy * dy3

        # m0 = m1 = m2 = m3 = 1
        mask = (m0 == 1) & (m1 == 1) & (m2 == 1) & (m3 == 1)
        w0[mask] += coef0[mask]
        w1[mask] += coef1[mask]
        w2[mask] += coef2[mask]
        w3[mask] += coef3[mask]

        # m3 = 0 normal=(1, 1)
        # qx * nx + qy * ny = 0
        # u3 * c3 = - (u0 * c0 + u1 * c1 + u2 * c2)
        mask = (m3 == 0) & (m2 != 0) & (m1 != 0) & (m0 != 0)
        nx, ny = 1, 1

        c0 = ((dxx * dx0 + dxy * dy0) * nx + (dyx * dx0 + dyy * dy0) * ny)
        c1 = ((dxx * dx1 + dxy * dy1) * nx + (dyx * dx1 + dyy * dy1) * ny)
        c2 = ((dxx * dx2 + dxy * dy2) * nx + (dyx * dx2 + dyy * dy2) * ny)
        c3 = ((dxx * dx3 + dxy * dy3) * nx + (dyx * dx3 + dyy * dy3) * ny)

        w0[mask] += coef0[mask] - coef3[mask] * c0[mask] / c3[mask]
        w1[mask] += coef1[mask] - coef3[mask] * c1[mask] / c3[mask]
        w2[mask] += coef2[mask] - coef3[mask] * c2[mask] / c3[mask]
        # w3[mask] = 0

        # m2 = 0 normal=(-1, 1)
        mask = (m2 == 0) & (m3 != 0) & (m1 != 0) & (m0 != 0)
        nx, ny = -1, 1
        c0 = ((dxx * dx0 + dxy * dy0) * nx + (dyx * dx0 + dyy * dy0) * ny)
        c1 = ((dxx * dx1 + dxy * dy1) * nx + (dyx * dx1 + dyy * dy1) * ny)
        c3 = ((dxx * dx3 + dxy * dy3) * nx + (dyx * dx3 + dyy * dy3) * ny)
        c2 = ((dxx * dx2 + dxy * dy2) * nx + (dyx * dx2 + dyy * dy2) * ny)

        w0[mask] += coef0[mask] - coef2[mask] * c0[mask] / c2[mask]
        w1[mask] += coef1[mask] - coef2[mask] * c1[mask] / c2[mask]
        # w2[mask] = 0
        w3[mask] += coef3[mask] - coef2[mask] * c3[mask] / c2[mask]

        # # m0 = 0 or m1 = 0
        # mask = ((m0 == 0) | (m1 == 0)) & (m2 != 0) & (m3 != 0)
        # w0[mask] = 0
        # w1[mask] = 0
        # w2[mask] = 0
        # w3[mask] = 0

        # m2 = m3 = 0 normal=(0, 1)
        # qy = dyx * dx + dyy * dy = 0 => dy = - (dyx / dyy) * dx
        # dx = u1 - u0
        # qx = (dxx - dxy * dyx / dyy) * dx
        mask = (m2 == 0) & (m3 == 0) & (m1 != 0) & (m0 != 0)
        w0[mask] = - (dxx - dxy * dyx / dyy)[mask]
        w1[mask] = (dxx - dxy * dyx / dyy)[mask]
        # w2[mask] = 0
        # w3[mask] = 0

        return w0, w1, w2, w3

    def flow_lower_component(self, d, m0, m1, m2, m3):
        """
        q = d @ [dx, dy]
        q @ n = 0, with n the normal vector at the boundary

        qx = dxx * dx + dxy * dy
        dx = 0.5 * (u1 + u3) - 0.5 * (u0 + u2)
        dy = 0.5 * (u0 + u1) - 0.5 * (u2 + u3)

        .. code-block:: text
            m0 ----- m1
            |        |
            |        |
            |        |
            m2 ----- m3
        """
        w0 = np.zeros_like(m0, dtype=d.dtype)
        w1 = np.zeros_like(m1, dtype=d.dtype)
        w2 = np.zeros_like(m2, dtype=d.dtype)
        w3 = np.zeros_like(m3, dtype=d.dtype)

        dxx = d[:, 0, 0]
        dxy = d[:, 0, 1]
        dyx = d[:, 1, 0]
        dyy = d[:, 1, 1]

        dx0, dy0 = - 0.5, 0.5
        dx1, dy1 = 0.5, 0.5
        dx2, dy2 = - 0.5, - 0.5
        dx3, dy3 = 0.5, - 0.5

        coef0 = dxx * dx0 + dxy * dy0
        coef1 = dxx * dx1 + dxy * dy1
        coef2 = dxx * dx2 + dxy * dy2
        coef3 = dxx * dx3 + dxy * dy3

        # m0 = m1 = m2 = m3 = 1
        mask = (m0 == 1) & (m1 == 1) & (m2 == 1) & (m3 == 1)
        w0[mask] += coef0[mask]
        w1[mask] += coef1[mask]
        w2[mask] += coef2[mask]
        w3[mask] += coef3[mask]

        # m3 = 0 normal=(1, -1)
        mask = (m3 == 0) & (m2 != 0) & (m1 != 0) & (m0 != 0)
        nx, ny = 1, -1

        c0 = ((dxx * dx0 + dxy * dy0) * nx + (dyx * dx0 + dyy * dy0) * ny)
        c1 = ((dxx * dx1 + dxy * dy1) * nx + (dyx * dx1 + dyy * dy1) * ny)
        c2 = ((dxx * dx2 + dxy * dy2) * nx + (dyx * dx2 + dyy * dy2) * ny)
        c3 = ((dxx * dx3 + dxy * dy3) * nx + (dyx * dx3 + dyy * dy3) * ny)

        w0[mask] += coef0[mask] - coef3[mask] * c0[mask] / c3[mask]
        w1[mask] += coef1[mask] - coef3[mask] * c1[mask] / c3[mask]
        w2[mask] += coef2[mask] - coef3[mask] * c2[mask] / c3[mask]
        # w3[mask] = 0

        # m2 = 0 normal=(-1, -1)
        mask = (m2 == 0) & (m3 != 0) & (m1 != 0) & (m0 != 0)
        nx, ny = -1, -1
        c0 = ((dxx * dx0 + dxy * dy0) * nx + (dyx * dx0 + dyy * dy0) * ny)
        c1 = ((dxx * dx1 + dxy * dy1) * nx + (dyx * dx1 + dyy * dy1) * ny)
        c3 = ((dxx * dx3 + dxy * dy3) * nx + (dyx * dx3 + dyy * dy3) * ny)
        c2 = ((dxx * dx2 + dxy * dy2) * nx + (dyx * dx2 + dyy * dy2) * ny)

        w0[mask] += coef0[mask] - coef2[mask] * c0[mask] / c2[mask]
        w1[mask] += coef1[mask] - coef2[mask] * c1[mask] / c2[mask]
        # w2[mask] = 0
        w3[mask] += coef3[mask] - coef2[mask] * c3[mask] / c2[mask]

        # # m0 = 0 or m1 = 0
        # mask = ((m0 == 0) | (m1 == 0)) & (m2 != 0) & (m3 != 0)
        # w0[mask] = 0
        # w1[mask] = 0
        # w2[mask] = 0
        # w3[mask] = 0

        # m2 = m3 = 0 normal=(0, -1)
        # qy = dyx * dx + dyy * dy = 0 => dy = - (dyx / dyy) * dx
        # dx = u1 - u0
        # qx = (dxx - dxy * dyx / dyy) * dx
        mask = (m2 == 0) & (m3 == 0) & (m1 != 0) & (m0 != 0)
        w0[mask] = - (dxx - dxy * dyx / dyy)[mask]
        w1[mask] = (dxx - dxy * dyx / dyy)[mask]
        # w2[mask] = 0
        # w3[mask] = 0

        return w0, w1, w2, w3
