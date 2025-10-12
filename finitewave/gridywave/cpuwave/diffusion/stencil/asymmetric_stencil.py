import numpy as np
from scipy import sparse as sp
from .stencil import Stencil


class AsymmetricStencil(Stencil):
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
        self.boundary = IsotropicFirstOrderBoundary()

    def assemble_matrices(self, simulation):
        """
        Computes the weights for isotropic diffusion in 2D.

        Parameters
        ----------
        simulation : simulation
            A simulation object containing the simulation parameters.

        Returns
        -------
        numpy.ndarray
            The weights for isotropic diffusion in 2D.
        """
        tissue = simulation.cardiac_tissue
        model = simulation.cardiac_model

        d_model = model.D_model
        mesh = tissue.mesh.copy()
        mesh[mesh != 1] = 0

        conductivity = tissue.conductivity
        conductivity *= np.ones_like(mesh, dtype=simulation.npfloat)
        diffusion = d_model * conductivity / simulation.dr**2

        stiff, mass = self.compute_weights_sparse(mesh, diffusion,
                                                  tissue.myo_indexes)
        return stiff, mass

    def compute_weights_sparse(self, mesh, diffusion, indexes):
        rows, cols, weights = self.boundary.compute(mesh, indexes)
        weights = weights.astype(diffusion.dtype)
        weights *= 0.5 * (diffusion.flat[cols] + diffusion.flat[rows])

        c_indexes = np.zeros(mesh.size, dtype=np.int64)
        c_indexes[indexes] = np.arange(len(indexes))
        rows = c_indexes[rows]
        cols = c_indexes[cols]

        size = len(indexes)
        shape = (size, size)
        K = sp.csr_matrix((weights, (rows, cols)), shape=shape)
        row_sums = np.array(K.sum(axis=1)).ravel()
        D = sp.diags(-row_sums, offsets=0, format='csr')
        K_new = K + D
        M = sp.diags(np.ones_like(row_sums), offsets=0, format='csr')
        return K_new.tocsr(), M.tocsr()


class AsymmetricFirstOrderBoundary:
    def __init__(self):
        pass

    def compute_weights(self, mesh, diffusion, indexes):
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
        res = self.valid_neighbors(major_axis, minor_axis, major_neighbor,
                                   mesh, indexes)
        neighbors_ijk, neighbors_m, major_mask = res
        center_ijk = neighbors_ijk[0]

        d_major = 0.5 * (d_major + np.roll(d_major, -major_neighbor)
                         )[major_mask]
        d_minor = 0.5 * (d_minor + np.roll(d_minor, -major_neighbor)
                         )[major_mask]

        weights = self.flow_weights(d_major, d_minor, *neighbors_m)

        rows = [np.ravel_multi_index(center_ijk[:, w != 0], mesh.shape)
                for w in weights]
        cols = [np.ravel_multi_index(ijk[:, w != 0], mesh.shape)
                for ijk, w in zip(neighbors_ijk, weights)]
        weights = [w[w != 0] for w in weights]

        return rows, cols, weights

    def valid_neighbors(self, major_axis, minor_axis, major_neighbor, mesh,
                        indexes):
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
        minor_lower = major_neighbor
        minor_upper = - major_neighbor
        res = self.valid_majors(indexes, mesh, major_axis, major_neighbor)
        ijk_center, ijk_major, m_center, m_major, major_mask = res

        ijk_minor_1, m_minor_1 = self.valid_minors(ijk_major, mesh,
                                                   minor_axis, minor_lower)
        ijk_minor_2, m_minor_2 = self.valid_minors(ijk_center, mesh,
                                                   minor_axis, minor_lower)
        ijk_minor_3, m_minor_3 = self.valid_minors(ijk_major, mesh,
                                                   minor_axis, minor_upper)
        ijk_minor_4, m_minor_4 = self.valid_minors(ijk_center, mesh,
                                                   minor_axis, minor_upper)

        neighbors_ijk = [ijk_center, ijk_major, ijk_minor_1, ijk_minor_2,
                         ijk_minor_3, ijk_minor_4]
        neighbors_m = [m_center, m_major, m_minor_1, m_minor_2,
                       m_minor_3, m_minor_4]
        return neighbors_ijk, neighbors_m, major_mask

    def valid_majors(self, indexes, mesh, major_axis, major_neighbor):
        """
        Selects valid major components for the stencil.

        .. code-block:: text
             major ------ d ---- center
        """
        ijk_center = np.array(np.unravel_index(indexes, mesh.shape))
        m_center = mesh[tuple(ijk_center)]

        ijk_major = ijk_center.copy()
        ijk_major[major_axis] += major_neighbor

        m_major = self.is_valid_neighbor(ijk_major, mesh, major_axis)

        major_mask = m_major > 0

        ijk_center = ijk_center[:, major_mask]
        m_center = m_center[major_mask]

        ijk_major = ijk_major[:, major_mask]
        m_major = m_major[major_mask]

        return ijk_center, ijk_major, m_center, m_major, major_mask

    def valid_minors(self, ijk, mesh, minor_axis, minor_neighbor):
        """
        Selects valid minor components for the stencil.
        """
        ijk_minor = ijk.copy()
        ijk_minor[minor_axis] += minor_neighbor
        m_minor = self.is_valid_neighbor(ijk_minor, mesh, minor_axis)
        return ijk_minor, m_minor

    def is_valid_neighbor(self, neighbor, mesh, i_axis):
        """
        Checks if the neighbor is valid
        (i.e., within the mesh and not empty).
        """
        mask = ((neighbor[i_axis] >= 0) &
                (neighbor[i_axis] < mesh.shape[i_axis]))
        mask[mask] = mesh[tuple(neighbor[:, mask])] > 0
        return mask.astype(mesh.dtype)

    def flow_weights(self, d_major, d_minor, m, m0, m1, m2, m3, m4):
        """
        Calculates the flow weights from m2 to m3 using an asymmetric stencil.

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

        w, w0, w1, w2, w3, w4 = self.minor_component(m, m0, m1, m2, m3, m4)

        w = - d_major * m0 + d_minor * w
        w0 = d_major * m0 + d_minor * w0
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

        mask = ((m == 0) | (m3 == 0) | (m_upper < 3) | (m_lower < 3))

        w = np.where(mask, 0, m / m_upper - m / m_lower)
        w0 = np.where(mask, 0, m0 / m_upper - m0 / m_lower)
        w1 = np.where(mask, 0, - m1 / m_lower)
        w2 = np.where(mask, 0, - m2 / m_lower)
        w3 = np.where(mask, 0, m3 / m_upper)
        w4 = np.where(mask, 0, m4 / m_upper)

        return w, w0, w1, w2, w3, w4
