import numpy as np
from .stencil import Stencil


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
        (out of bounds or in an empty cell), flux from this neighbor is zero.
    - If more than one minor neighbor from upper or lower side is invalid,
        flux from these neighbors is determined only by the major component.
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

    def compute_diffusion_weights(self, mesh, diffusion, dr, indexes):
        """
        Computes the weights for the isotropic stencil with first-order
        boundary conditions.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        diffusion : numpy.ndarray
            The diffusion tensor as a (*mesh.shape, ndim, ndim).
        dr : float
            The grid spacing.
        indexes : numpy.ndarray
            The indexes of the non-empty points in the mesh.

        Returns
        -------
        rows : numpy.ndarray
            The central node indexes.
        cols : numpy.ndarray
            The node indexes involved in diffusion calculation.
        weights : numpy.ndarray
            The weights for connections between central and neighboring nodes.
        """
        rows = []
        cols = []
        weights = []

        ijk = np.array(np.unravel_index(indexes, mesh.shape))
        for axis in range(mesh.ndim):
            r, c, w = self.compute_diffusion_along_axis(mesh, diffusion, dr, ijk, axis)
            rows.append(r)
            cols.append(c)
            weights.append(w)

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights)
        return rows, cols, weights
    
    def compute_diffusion_along_axis(self, mesh, diffusion, dr, ijk, axis):
        """
        Computes the diffusion weights along a given axis.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        dr : float
            The grid spacing.
        ijk : numpy.ndarray
            The indexes of the non-empty points in the mesh.
        ijk_major : numpy.ndarray
            The indexes of the major neighbor points.
        ijk_list : list
            A list of numpy arrays containing the indexes of the involved points.
        w_list : list
            A list of numpy arrays containing the flux weights for the involved points.

        Returns
        -------
        rows : np.ndarray
            The central node indexes.
        cols : np.ndarray
            The node indexes involved in diffusion calculation.
        weights : np.ndarray
            The weights for connections.
        """
        
        ijk_major, ijk_list, w_list = self.compute_flux_weights(mesh, diffusion, dr, ijk, axis)
        major_to_center = self.nonzero_weights(mesh, ijk, ijk_list, w_list)
        center_to_major = self.nonzero_weights(mesh, ijk_major, ijk_list, w_list, direction=-1)

        rows = np.concatenate(major_to_center[0] + center_to_major[0])
        cols = np.concatenate(major_to_center[1] + center_to_major[1])
        weights = np.concatenate(major_to_center[2] + center_to_major[2]) / dr
        return rows, cols, weights
    
    def compute_flux_weights(self, mesh, diffusion, dr, ijk, major_axis):
        """
        Computes the flux weights along major axis.

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
        dr : float
            The grid spacing.
        ijk : numpy.ndarray
            The indexes of the non-empty cells in the mesh.
        major_axis : int
            The axis of the major direction.

        Returns 
        -------
        ijk_major : numpy.ndarray
            The indexes of the major neighbor cells.
        ijk_list : list
            The list of coordinates of the involved nodes in the mesh.
        w_list : list
            The list of weights for the involved nodes.
        """
        ijk_major = self.build_neighbor(ijk, shift=1, axis=major_axis)
        m_center = self.is_valid_index(ijk, mesh).astype(diffusion.dtype)
        m_major = self.is_valid_index(ijk_major, mesh).astype(diffusion.dtype)
        
        ijk_list, w_list = self.major_flux_weights(diffusion, dr,
                                                    ijk, ijk_major,
                                                    m_center, m_major,
                                                    major_axis)

        minor_axes = np.roll(np.arange(mesh.ndim), -major_axis)[1:]

        for minor_axis in minor_axes:
            ijk_minors, w_minors = self.minor_flux_weights(mesh, diffusion, dr,
                                                           ijk, ijk_major,
                                                           m_center, m_major,
                                                           major_axis, minor_axis)
            ijk_list += ijk_minors
            w_list += w_minors

        return ijk_major, ijk_list, w_list
    
    def major_flux_weights(self, diffusion, dr, ijk, ijk_major, m_center, m_major, major_axis):
        """Computes the weights for the major component of the flux from
        major to center.

        q_x = - Dxx * (u_major - u_center) / dr
        
        Parameters
        ----------
        diffusion : numpy.ndarray
            The diffusion tensor as a (*mesh.shape, ndim, ndim).
        dr : float
            The grid spacing.
        ijk : numpy.ndarray
            The indexes of the non-empty cells in the mesh.
        ijk_major : numpy.ndarray
            The indexes of the major neighbor cells.
        m_center : numpy.ndarray
            The validity mask of the center cell.
        m_major : numpy.ndarray
            The validity mask of the major neighbor.
        major_axis : int
            The axis of the major direction.

        Returns
        -------
        tuple
            A tuple containing the ijk coordinates of the involved cells and
            their flux weights.
        """
        valid_connection = (m_major > 0) & (m_center > 0)
        w_major = - self.diffusion_component(diffusion, ijk, major_axis,
                                             major_axis, valid_connection) / dr

        ijk_list = [ijk, ijk_major]
        w_list = [-w_major, w_major]
        return ijk_list, w_list

    def minor_flux_weights(self, mesh, diffusion, dr, ijk_center, ijk_major,
                           m_center, m_major, major_axis, minor_axis):
        """
        Calculates the minor flux weights.

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
        dr : float
            The grid spacing.
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
            their flux weights.

        Notes
        -----
        The diffusion components are calculated in the middle of central and
        major nodes, therefore the diffusion coefficient is averaged between
        the corresponding nodes.
        """

        d_minor = self.diffusion_component(diffusion, ijk_center, major_axis, minor_axis, m_major)
        d_major = self.diffusion_component(diffusion, ijk_center, minor_axis, minor_axis, m_major)

        ijk_1 = self.build_neighbor(ijk_center, -1, minor_axis)
        ijk_2 = self.build_neighbor(ijk_major, -1, minor_axis)
        ijk_3 = self.build_neighbor(ijk_center, 1, minor_axis)
        ijk_4 = self.build_neighbor(ijk_major, 1, minor_axis)

        m1 = self.is_valid_index(ijk_1, mesh).astype(diffusion.dtype)
        m2 = self.is_valid_index(ijk_2, mesh).astype(diffusion.dtype)
        m3 = self.is_valid_index(ijk_3, mesh).astype(diffusion.dtype)
        m4 = self.is_valid_index(ijk_4, mesh).astype(diffusion.dtype)

        weights = self.minor_component(d_minor, d_major, dr, m_center, m_major, m1, m2, m3, m4)
        ijk_list = [ijk_center, ijk_major, ijk_1, ijk_2, ijk_3, ijk_4]
        return ijk_list, weights

    def diffusion_component(self, diffusion, ijk, axis1, axis2, mask):
        """
        Averages the diffusion coefficients between two sets of cells.

        Parameters
        ----------
        diffusion : numpy.ndarray [*mesh.shape, ndim, ndim]
            The diffusion tensor at the center of connections between nodes.
        ijk : numpy.ndarray
            The indexes of the central nodes.
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
        d[mask > 0] = diffusion[*ijk[:, mask > 0], axis1, axis2]
        return d

    def major_component(self, d_major, dr, m_center, m_major):
        """
        Calculates the major component of the flux.

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
        dr : float
            The grid spacing.
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
        w_major = d_major * m_major * m_center / dr

        return w_major

    def minor_component(self, d_minor, d_major, dr, m, m0, m1, m2, m3, m4):
        """
        Calculates the minor component of the flux.

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
        dr : float
            The grid spacing.
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

        # mask = ((m == 0) | (m0 == 0) | (m_upper < 3) | (m_lower < 3))
        # more stable version, but less precise
        mask = ((m == 0) | (m0 == 0) | (m_upper < 4) | (m_lower < 4))

        w = d_minor / dr * np.where(mask, 0, m / m_upper - m / m_lower)
        w0 = d_minor / dr * np.where(mask, 0, m0 / m_upper - m0 / m_lower)
        w1 = d_minor / dr * np.where(mask, 0, - m1 / m_lower)
        w2 = d_minor / dr * np.where(mask, 0, - m2 / m_lower)
        w3 = d_minor / dr * np.where(mask, 0, m3 / m_upper)
        w4 = d_minor / dr * np.where(mask, 0, m4 / m_upper)
       
        # mask = ((m == 1) & (m0 == 1) & ((m1 + m2 + m3 + m4) < 4))
        # w[mask] = - d_minor[mask] / d_major[mask]
        # w0[mask] = d_minor[mask] / d_major[mask]

        return w, w0, w1, w2, w3, w4
