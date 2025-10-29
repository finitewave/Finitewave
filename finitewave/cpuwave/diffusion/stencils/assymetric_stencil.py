import numpy as np


class AsymmetricStencil:
    """
    This class computes indexes and weights for the asymmetric stencil.

    Notes
    -----
    The asymmetric stencil reduces to the isotropic stencil with first-order
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
        ijk = np.array(np.unravel_index(indexes, mesh.shape))
        for i in range(mesh.ndim):
            axis = np.roll(np.arange(mesh.ndim), i)
            major_axis = axis[0]
            minor_axes = axis[1:]
            for minor_axis in minor_axes:
                d_major = diffusion[*ijk, major_axis, major_axis]
                d_minor = diffusion[*ijk, major_axis, minor_axis]
                for major_neighbor in [-1, 1]:
                    res = self.compute_flow_weights(d_major, d_minor,
                                                    major_axis, minor_axis,
                                                    major_neighbor, mesh,
                                                    indexes)
                    rows += res[0]
                    cols += res[1]
                    weights += res[2]

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights)
        return rows, cols, weights

    def compute_flow_weights(self, d_major, d_minor, major_axis, minor_axis,
                             major_neighbor, mesh, indexes):
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
        ijk_center = ijk_list[0]

        d_major = 0.5 * (d_major + np.roll(d_major, -major_neighbor))
        d_minor = 0.5 * (d_minor + np.roll(d_minor, -major_neighbor))

        weights = self.flow_weights(d_major, d_minor, *mask_list)

        rows = [np.ravel_multi_index(ijk_center[:, w != 0], mesh.shape)
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
              major ------ d ---- center
                |                   |
                |                   |
                |                   |
             minor_1 ------------ minor_2
        """
        minor_lower = major_shift
        minor_upper = - major_shift

        ijk_center = np.array(np.unravel_index(indexes, mesh.shape))
        ijk_major = self.build_neighbor(ijk_center, major_shift, major_axis)
        ijk_minor_1 = self.build_neighbor(ijk_major, minor_lower, minor_axis)
        ijk_minor_2 = self.build_neighbor(ijk_center, minor_lower, minor_axis)
        ijk_minor_3 = self.build_neighbor(ijk_major, minor_upper, minor_axis)
        ijk_minor_4 = self.build_neighbor(ijk_center, minor_upper, minor_axis)

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

    def flow_weights(self, d_major, d_minor, m_center, m_major, m_minor_1,
                     m_minor_2, m_minor_3, m_minor_4):
        """
        Calculates the flow weights from m2 to m3 using an asymmetric stencil.

        .. code-block:: text
            minor_3 ------------ minor_4
                |                   |
                |                   |
                |                   |
              major ------ d ---- center
                |                   |
                |                   |
                |                   |
             minor_1 ------------ minor_2
        """

        w, w0, w1, w2, w3, w4 = self.minor_component(m_center, m_major,
                                                     m_minor_1, m_minor_2,
                                                     m_minor_3, m_minor_4)

        w = - d_major * m_major + d_minor * w
        w0 = d_major * m_major + d_minor * w0
        w1 *= d_minor
        w2 *= d_minor

        w3 *= d_minor
        w4 *= d_minor

        return w, w0, w1, w2, w3, w4

    def minor_component(self, m, m0, m1, m2, m3, m4):
        """
        Calculates the minor component of the flow.

        .. code-block:: text
            m3 ----- m4
            |        |
            |        |
            |        |
            m0 - d - m
            |        |
            |        |
            |        |
            m1 ----- m2
        """
        m_upper = m3 + m4 + m + m0
        m_lower = m1 + m2 + m + m0

        mask = ((m == 0) | (m0 == 0) | (m_upper < 3) | (m_lower < 3))

        w = np.where(mask, 0, m / m_upper - m / m_lower)
        w0 = np.where(mask, 0, m0 / m_upper - m0 / m_lower)
        w1 = np.where(mask, 0, - m1 / m_lower)
        w2 = np.where(mask, 0, - m2 / m_lower)
        w3 = np.where(mask, 0, m3 / m_upper)
        w4 = np.where(mask, 0, m4 / m_upper)

        return w, w0, w1, w2, w3, w4