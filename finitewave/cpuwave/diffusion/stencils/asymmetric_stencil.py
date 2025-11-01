import numpy as np
from .stecil import Stencil


class AsymmetricStencil(Stencil):
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
        super().__init__()

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

        Returns
        -------
        tuple
            A tuple containing the rows, cols, and weights for the stencil.
        """
        rows, cols, weights = [], [], []
        ijk = np.array(np.unravel_index(indexes, mesh.shape))
        for i in range(mesh.ndim):
            axis = np.roll(np.arange(mesh.ndim), i)
            major_axis = axis[0]
            minor_axes = axis[1:]

            res = self.compute_flow_weights(mesh, diffusion, ijk, major_axis,
                                            minor_axes)
            rows += res[0]
            cols += res[1]
            weights += res[2]

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights)
        return rows, cols, weights

    def compute_flow_weights(self, mesh, diffusion, ijk, major_axis,
                             minor_axes):
        """
        Computes the flow weights for the asymmetric stencil.

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

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        diffusion : numpy.ndarray
            The diffusion tensor as a (*mesh.shape, ndim, ndim).
        ijk : numpy.ndarray
            The indexes of the non-empty cells in the mesh.
        major_axis : int
            The axis of the major direction.
        minor_axes : list
            The axes of the minor directions.

        Returns
        -------
        tuple
            A tuple containing the rows, cols, and weights for the flow.
        """
        res = self.major_flow_weights(mesh, diffusion, ijk, major_axis)
        ijk_center, ijk_major, m_center, m_major, w_center, w_major = res
        ijk_list = [ijk_center, ijk_major]
        w_list = [w_center, w_major]

        # collect major flow weights
        rows, cols, weights = self.nonzero_weights(mesh, ijk_center, ijk_major,
                                                   ijk_list, w_list)

        for minor_axis in minor_axes:
            ijk_list, w_list = self.minor_flow_weights(mesh, diffusion,
                                                       ijk_center, ijk_major,
                                                       m_center, m_major,
                                                       major_axis, minor_axis)
            r, c, w = self.nonzero_weights(mesh, ijk_center, ijk_major,
                                           ijk_list, w_list)
            rows += r
            cols += c
            weights += w
        return rows, cols, weights

    def major_flow_weights(self, mesh, diffusion, ijk, major_axis):
        """
        Calculates major component of the flow from center to major neighbor.

        qx = Dxx * (du/dx)
           = Dxx * (major - center)

        .. code-block:: text
             minor_3 ----------- minor_4
                |                   |
                |                   |
                |                   |
              center ---- d ---- major
                |                   |
                |                   |
                |                   |
             minor_1 ----------- minor_2

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        ijk : numpy.ndarray
            The indexes of the non-empty cells in the mesh.
        d_major : numpy.ndarray
            The major diffusion coefficients.
        major_axis : int
            The axis of the major direction.
        major_shift : int
            The shift in the major direction.

        Returns
        -------
        tuple
            A tuple containing the ijk coordinates of the center and major
            neighbors, their validity masks, and the flow weights.

        Notes
        -----
        The diffusion components are calculated in the middle of two nodes,
        therefore the diffusion coefficient is averaged between the
        corresponding nodes.
        """
        ijk_center = ijk
        m_center = self.is_valid_neighbor(ijk_center, mesh)

        ijk_major = self.build_neighbor(ijk_center, 1, major_axis)
        m_major = self.is_valid_neighbor(ijk_major, mesh)

        d_major = self.average_diffusion(diffusion, ijk_center, ijk_major,
                                         major_axis, major_axis, m_major)

        w_center, w_major = self.major_component(d_major, m_center, m_major)
        return ijk_center, ijk_major, m_center, m_major, w_center, w_major

    def minor_flow_weights(self, mesh, diffusion, ijk_center, ijk_major,
                           m_center, m_major, major_axis, minor_axis):
        """
        Calculates the minor flow weights.

        qy = Dxy * (du/dy)
           = Dxy * ((minor_3 + minor_4 + center + major) / 4 -
                    (minor_1 + minor_2 + center + major) / 4)

        .. code-block:: text
             minor_3 ----------- minor_4
                |                   |
                |                   |
                |                   |
              center ---- d ---- major
                |                   |
                |                   |
                |                   |
             minor_1 ----------- minor_2

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        diffusion : numpy.ndarray
            The diffusion tensor as a (*mesh.shape, ndim, ndim).
        ijk_center : numpy.ndarray
            The indexes of the center cells.
        ijk_major : numpy.ndarray
            The indexes of the major neighbor cells.
        m_center : numpy.ndarray
            The validity mask of the center cell.
        m_major : numpy.ndarray
            The validity mask of the major neighbor.
        major_shift : int
            The shift in the major direction.
        minor_axis : int
            The axis of the minor direction.

        Returns
        -------
        tuple
            A tuple containing the ijk coordinates of the involved cells and
            their flow weights.

        Notes
        -----
        The diffusion components are calculated in the middle of central and
        major nodes, therefore the diffusion coefficient is averaged between
        the corresponding nodes.
        """

        d_minor = self.average_diffusion(diffusion, ijk_center, ijk_major,
                                         major_axis, minor_axis, m_major)

        ijk_1 = self.build_neighbor(ijk_center, -1, minor_axis)
        ijk_2 = self.build_neighbor(ijk_major, -1, minor_axis)
        ijk_3 = self.build_neighbor(ijk_center, 1, minor_axis)
        ijk_4 = self.build_neighbor(ijk_major, 1, minor_axis)

        m1 = self.is_valid_neighbor(ijk_1, mesh)
        m2 = self.is_valid_neighbor(ijk_2, mesh)
        m3 = self.is_valid_neighbor(ijk_3, mesh)
        m4 = self.is_valid_neighbor(ijk_4, mesh)

        weights = self.minor_component(d_minor, m_center, m_major, m1, m2, m3, m4)
        ijk_list = [ijk_center, ijk_major, ijk_1, ijk_2, ijk_3, ijk_4]
        return ijk_list, weights

    def average_diffusion(self, diffusion, ijk_1, ijk_2, axis1, axis2, mask):
        """
        Averages the diffusion coefficients between two sets of cells.

        Parameters
        ----------
        diffusion : numpy.ndarray
            The diffusion tensor as a (*mesh.shape, ndim, ndim).
        ijk_1 : numpy.ndarray
            The indexes of the first set of cells.
        ijk_2 : numpy.ndarray
            The indexes of the second set of cells.
        axis1 : int
            The first axis index.
        axis2 : int
            The second axis index.
        mask : numpy.ndarray
            The validity mask of the cells.

        Returns
        -------
        numpy.ndarray
            The averaged diffusion coefficients.
        """
        d = np.zeros(mask.shape, dtype=diffusion.dtype)
        d[mask > 0] = 0.5 * (diffusion[*ijk_1[:, mask > 0], axis1, axis2] +
                             diffusion[*ijk_2[:, mask > 0], axis1, axis2])
        return d

    def major_component(self, d_major, m_center, m_major):
        """
        Calculates the major component of the flow.

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

        Parameters
        ----------
        d_major : numpy.ndarray
            The major diffusion coefficients.
        m_center : numpy.ndarray
            The validity mask of the center cell.
        m_major : numpy.ndarray
            The validity mask of the major neighbor.

        Returns
        -------
        tuple
            A tuple containing the weights for the center and major neighbors.
        """
        # d_major = 0.5 * (d_major + np.roll(d_major, major_shift))
        w_center = - d_major * m_major * m_center
        w_major = d_major * m_major * m_center

        return w_center, w_major

    def minor_component(self, d_minor, m, m0, m1, m2, m3, m4):
        """
        Calculates the minor component of the flow.

        .. code-block:: text
            m3 ----- m4
            |        |
            |        |
            |        |
            m - d -  m0
            |        |
            |        |
            |        |
            m1 ----- m2

        Parameters
        ----------
        d_minor : numpy.ndarray
            The minor diffusion coefficients.
        m : numpy.ndarray
            The validity mask of the center cell.
        m0 : numpy.ndarray
            The validity mask of the major neighbor.
        m1 : numpy.ndarray
            The validity mask of minor neighbor 1.
        m2 : numpy.ndarray
            The validity mask of minor neighbor 2.
        m3 : numpy.ndarray
            The validity mask of minor neighbor 3.
        m4 : numpy.ndarray
            The validity mask of minor neighbor 4.

        Returns
        -------
        tuple
            A tuple containing the weights for the center, major, and minor
            neighbors.
        """
        # d_minor = 0.5 * (d_minor + np.roll(d_minor, major_shift))
        m_upper = m3 + m4 + m + m0
        m_lower = m1 + m2 + m + m0

        mask = ((m == 0) | (m0 == 0) | (m_upper < 3) | (m_lower < 3))
        # more stable version, but less precise
        # mask = ((m == 0) | (m0 == 0) | (m_upper < 4) | (m_lower < 4))

        w = d_minor * np.where(mask, 0, m / m_upper - m / m_lower)
        w0 = d_minor * np.where(mask, 0, m0 / m_upper - m0 / m_lower)
        w1 = d_minor * np.where(mask, 0, - m1 / m_lower)
        w2 = d_minor * np.where(mask, 0, - m2 / m_lower)
        w3 = d_minor * np.where(mask, 0, m3 / m_upper)
        w4 = d_minor * np.where(mask, 0, m4 / m_upper)

        return w, w0, w1, w2, w3, w4
