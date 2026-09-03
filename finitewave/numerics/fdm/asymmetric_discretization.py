import numpy as np
from scipy import sparse
from .finite_difference_discretization import FiniteDifferenceDiscretization


class AsymmetricDiscretization(FiniteDifferenceDiscretization):
    """
    This class computes diffusion operator for the asymmetric finite difference stencil.

    Notes
    -----
    The asymmetric stencil reduces to the isotropic stencil with first-order
    boundary conditions if the ``fibers = None`` or ``fibers`` are aligned with
    the grid and ``D_al = D_ac``.

    Rules for handling boundaries are:
    - If a major (directly adjacent) neighbor is invalid
        (out of bounds or in an empty cell), flux from this neighbor is zero.
    - If any of the minor neighbor from upper or lower side is invalid,
        the minor flux from this axis is zero.

    Diffusion components are calculated in the middle of two nodes, therefore
    the diffusion coefficient is averaged between the corresponding nodes.

    References
    ----------
    Bram van Es, Barry Koren, Hugo J. de Blank,
    Finite-difference schemes for anisotropic diffusion,
    Journal of Computational Physics,
    Volume 272, 2014, Pages 526-549, ISSN 0021-9991,
    https://doi.org/10.1016/j.jcp.2014.04.046
    """
    def __init__(self, averaging_method="arithmetic"):
        """
        Initializes the AsymmetricDiscretization class.
        
        Parameters
        ----------
        averaging_method : str, optional
            The method to average diffusion coefficients. Options are "arithmetic" or "harmonic".
        """
        super().__init__()
        if averaging_method == "arithmetic":
            self.diffusion_averaging_method = lambda d1, d2: 0.5 * (d1 + d2)
        elif averaging_method == "harmonic":
            self.diffusion_averaging_method = lambda d1, d2: 2 * (d1 * d2) / (d1 + d2)
        else:
            raise ValueError(f"Invalid averaging method: {averaging_method}. "
                             "Choose 'arithmetic' or 'harmonic'.")

    def compute_diffusion_operator(self, mesh, dr, indexes=None, diffusion=1., connectivity=1.):
        """
        Builds the diffusion operator as sparse matrix with the asymmetric stencil.

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
        scipy.sparse.csr_matrix
            The diffusion operator as a sparse matrix.
        """
        rows = []
        cols = []
        weights = []

        if indexes is None:
            indexes = np.arange(mesh[mesh == 1].size, dtype=np.int64)

        tissue_size = mesh[mesh > 0].size

        tissue_index_map = - np.ones(mesh.shape, dtype=indexes.dtype)
        tissue_index_map[mesh > 0] = np.arange(tissue_size, dtype=indexes.dtype)

        ijk = np.array(np.unravel_index(indexes, mesh.shape))

        for axis in range(mesh.ndim):
            r, c, w = self._diffusion_operator_component(mesh, diffusion, connectivity, dr, ijk, axis, tissue_index_map)
            rows.append(r)
            cols.append(c)
            weights.append(w)

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights)
        return sparse.csr_matrix((weights, (rows, cols)), shape=(tissue_size, tissue_size))
    
    def _diffusion_operator_component(self, mesh, diffusion, connectivity, dr, ijk, axis, tissue_index_map):
        """
        Computes the diffusion weights along a given axis.

        - (q_x1 - q_x0) / dr

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
        
        ijk_major, ijk_list, w_list = self._flux_weights(mesh, diffusion, connectivity, dr, ijk, axis, tissue_index_map)
        major_to_center = self.nonzero_weights(mesh, ijk, ijk_list, w_list, tissue_index_map, direction=1)
        center_to_major = self.nonzero_weights(mesh, ijk_major, ijk_list, w_list, tissue_index_map, direction=-1)

        rows = np.concatenate([major_to_center[0], center_to_major[0]])
        cols = np.concatenate([major_to_center[1], center_to_major[1]])
        weights = np.concatenate([major_to_center[2], center_to_major[2]]) / dr
        return rows, cols, weights
    
    def _flux_weights(self, mesh, diffusion, connectivity, dr, ijk, major_axis, tissue_index_map):
        """
        Computes the flux weights for

        .. code-block:: text
            minor_3 ---- minor_4
              |            |
            center - d - major
              |            |
            minor_1 ---- minor_2

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
        m_major = self.is_valid_index(ijk_major, mesh)

        diffusion_tensor_component = self._diffusion_tensor_component(
            diffusion, connectivity, ijk, ijk_major, m_major, major_axis, tissue_index_map
        )

        diffusion_major = diffusion_tensor_component[:, major_axis]
        ijk_list, w_list = self._major_flux_weights(diffusion_major, dr, ijk, ijk_major, m_major)

        minor_axes = np.roll(np.arange(mesh.ndim), -major_axis)[1:]

        for minor_axis in minor_axes:
            diffusion_minor = diffusion_tensor_component[:, minor_axis]
            ijk_minors, w_minors = self._minor_flux_weights(diffusion_minor, dr, ijk, ijk_major, m_major, minor_axis, mesh)
            ijk_list += ijk_minors
            w_list += w_minors

        return ijk_major, ijk_list, w_list
    
    def _major_flux_weights(self, diffusion_major, dr, ijk, ijk_major, m_major):
        """Computes the weights for the major component of the flux from
        major to center.

        q_x = - Dxx * (u_major - u_center) / dr
        
        Parameters
        ----------
        diffusion_major : numpy.ndarray
            The diffusion coefficient along the major axis.
        dr : float
            The grid spacing.
        ijk : numpy.ndarray
            The indexes of the non-empty cells in the mesh.
        ijk_major : numpy.ndarray
            The indexes of the major neighbor cells.
        m_major : numpy.ndarray
            The validity mask of the major neighbor.

        Returns
        -------
        tuple
            A tuple containing the ijk coordinates of the involved cells and
            their flux weights.
        """
        w_major = np.where(m_major > 0, diffusion_major / dr, 0.)

        ijk_list = [ijk, ijk_major]
        w_list = [w_major, -w_major]
        return ijk_list, w_list

    def _minor_flux_weights(self, diffusion_minor, dr, ijk_center, ijk_major,
                           m_major, minor_axis, mesh):
        """
        Calculates the minor flux weights.

        qy = - Dxy * (du/dy)
           = - Dxy * ((u_3 + u_4) / 4 - (u_1 + u_2) / 4)

        .. code-block:: text
            minor_3 ---- minor_4
              |            |
            center - d - major
              |            |
            minor_1 ---- minor_2

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        diffusion_minor : numpy.ndarray
            The diffusion coefficient along the minor axis.
        dr : float
            The grid spacing.
        ijk_center : numpy.ndarray
            The indexes of the center cells.
        ijk_major : numpy.ndarray
            The indexes of the major neighbor cells.
        m_major : numpy.ndarray
            The validity mask of the major neighbor.
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
        ijk_1 = self.build_neighbor(ijk_center, -1, minor_axis)
        ijk_2 = self.build_neighbor(ijk_major, -1, minor_axis)
        ijk_3 = self.build_neighbor(ijk_center, 1, minor_axis)
        ijk_4 = self.build_neighbor(ijk_major, 1, minor_axis)

        m1 = self.is_valid_index(ijk_1, mesh)
        m2 = self.is_valid_index(ijk_2, mesh)
        m3 = self.is_valid_index(ijk_3, mesh)
        m4 = self.is_valid_index(ijk_4, mesh)

        mask = (m_major > 0) & (m1 > 0) & (m2 > 0) & (m3 > 0) & (m4 > 0)

        w = np.where(mask, diffusion_minor / (4 * dr), 0.)
        ijk_list = [ijk_1, ijk_2, ijk_3, ijk_4]
        w_list = [w, w, -w, -w]
        return ijk_list, w_list

    def _diffusion_tensor_component(self, diffusion, connectivity, ijk, ijk_major, 
                                   mask_major, major_axis, tissue_index_map):
        diffusion_along_major = self._average_diffusion_component(
            diffusion, ijk, ijk_major, mask_major, major_axis, tissue_index_map)
        connectivity_along_major = self._connectivity_component(
            connectivity, ijk, mask_major, major_axis, tissue_index_map)
        return diffusion_along_major * connectivity_along_major[:, None]

    def _average_diffusion_component(self, diffusion, ijk, ijk_neighbor, mask, axis, tissue_index_map):
        diffusion = np.atleast_1d(diffusion)
        ndim, n_points = ijk_neighbor.shape
        if diffusion.size == 1:
            return diffusion * np.eye(ndim)[axis]

        center_indexes = tissue_index_map[*ijk[:, mask > 0]]
        neighbor_indexes = tissue_index_map[*ijk_neighbor[:, mask > 0]]
        d_axis = self.diffusion_averaging_method(
            diffusion[center_indexes, axis, :], diffusion[neighbor_indexes, axis, :])

        d_full = np.zeros((n_points, ndim))
        d_full[mask > 0] = d_axis
        return d_full

    def _connectivity_component(self, connectivity, ijk, mask, axis, tissue_index_map):
        """
        Computes the connectivity along a given axis.

        Parameters
        ----------
        connectivity : scalar or numpy.ndarray
            The connectivity values.
        ijk : numpy.ndarray
            The indexes of the central nodes.
        mask : numpy.ndarray
            The validity mask of the cells.
        axis : int
            The axis along which to compute the connectivity.

        Returns
        -------
        numpy.ndarray
            The connectivity along the specified axis.
        """
        connectivity = np.asarray(connectivity)
        ndim, n_points = ijk.shape
        center_indexes = tissue_index_map[*ijk[:, mask > 0]]
    
        if connectivity.size == 1:
            return np.atleast_1d(connectivity)
        
        if connectivity.size == ndim:
            return np.atleast_1d(connectivity[axis])
        
        connectivity_along_axis = connectivity[:, axis].copy()
        connectivity_along_axis[center_indexes] = 0.
        return connectivity_along_axis
