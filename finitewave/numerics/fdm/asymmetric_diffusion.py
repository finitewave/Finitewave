import numpy as np
from scipy import sparse
from ._grid_utils import nonzero_weights, build_neighbor, is_valid_index


class AsymmetricDiffusion:
    """Assemble anisotropic diffusion using the asymmetric flux stencil.

    Parameters
    ----------
    averaging_method : {"arithmetic", "harmonic"}, optional
        Componentwise face averaging, default "arithmetic". Harmonic
        averaging requires nonzero sums of paired tensor components.

    Notes
    -----
    Valid neighbors are in bounds with mesh == 1. Missing major neighbors
    produce zero normal flux. Transverse flux requires the major neighbor
    and all four transverse neighbors to be active. Face contributions are
    added with opposite signs to their two endpoint rows.

    Scalar diffusion removes transverse terms. Its boundary treatment is
    not the reflected boundary stencil implemented by IsotropicDiffusion.

    References
    ----------
    B. van Es, B. Koren, H. J. de Blank, Finite-difference schemes for
    anisotropic diffusion, Journal of Computational Physics 272 (2014),
    526-549. DOI: 10.1016/j.jcp.2014.04.046.
    """
    def __init__(self, averaging_method="arithmetic"):
        """Select the face-averaging rule.

        Parameters
        ----------
        averaging_method : {"arithmetic", "harmonic"}, optional
            Componentwise averaging rule for diffusion at cell faces.
            Harmonic averaging requires nonzero sums of paired components. Default is 'arithmetic'.
        """
        if averaging_method == "arithmetic":
            self.diffusion_averaging_method = lambda d1, d2: 0.5 * (d1 + d2)
        elif averaging_method == "harmonic":
            self.diffusion_averaging_method = lambda d1, d2: 2 * (d1 * d2) / (d1 + d2)
        else:
            raise ValueError(f"Invalid averaging method: {averaging_method}. "
                             "Choose 'arithmetic' or 'harmonic'.")

    def build_diffusion_operator(self, mesh, dr, indexes=None, diffusion=1., connectivity=1.):
        """Build the selected diffusion stiffness matrix.

        Parameters
        ----------
        mesh : numpy.ndarray
            Grid labels: 0 for empty, 1 for active, 2 for fibrotic nodes.
            All mesh > 0 nodes are included in compact C-order matrix indexing.
        dr : float
            Uniform grid spacing along every axis.
        indexes : numpy.ndarray, optional
            Flat full-grid indices of center nodes. For operator builders,
            None selects mesh == 1; these are not compact matrix indices. Default is None.
        diffusion : float or numpy.ndarray, optional
            Isotropic coefficient or tensors of shape (N_tissue, ndim, ndim),
            ordered by flat indices where mesh > 0. N_tissue counts nonempty nodes. Default is 1.0.
        connectivity : float or numpy.ndarray, optional
            Connection factors: scalar, one per axis, or an array with shape
            (N_tissue, ndim), passed to the stencil connectivity helper. Default is 1.0.

        Returns
        -------
        stiffness : scipy.sparse.csr_matrix
            Matrix of shape (N_tissue, N_tissue), representing negative diffusion
            in du/dt = -K u. It includes the 1/dr**2 scaling.
        """
        rows = []
        cols = []
        weights = []

        if indexes is None:
            indexes = np.flatnonzero(mesh == 1)

        tissue_size = np.count_nonzero(mesh > 0)

        tissue_index_map = - np.ones(mesh.shape, dtype=indexes.dtype)
        tissue_index_map[mesh > 0] = np.arange(tissue_size, dtype=indexes.dtype)

        if np.any(tissue_index_map.flat[indexes] < 0):
            raise ValueError("Tissue index mapping failed. Check the mesh and indexes.")

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
        """Assemble one axis contribution to the stiffness matrix.

        Parameters
        ----------
        mesh : numpy.ndarray
            Grid labels: 0 for empty, 1 for active, 2 for fibrotic nodes.
            All mesh > 0 nodes are included in compact C-order matrix indexing.
        diffusion : float or numpy.ndarray
            Isotropic coefficient or tensors of shape (N_tissue, ndim, ndim),
            ordered by flat indices where mesh > 0. N_tissue counts nonempty nodes.
        connectivity : float or numpy.ndarray
            Connection factors: scalar, one per axis, or an array with shape
            (N_tissue, ndim), passed to the stencil connectivity helper.
        dr : float
            Uniform grid spacing along every axis.
        ijk : numpy.ndarray, shape (ndim, N_points)
            Full-grid coordinates of center nodes.
        axis : int
            Grid axis.
        tissue_index_map : numpy.ndarray, shape mesh.shape
            Map full-grid coordinates to compact nonempty-node indices.

        Returns
        -------
        rows : numpy.ndarray
            Compact matrix row indices.
        cols : numpy.ndarray
            Compact matrix column indices.
        weights : numpy.ndarray
            Corresponding matrix coefficients.
        """

        ijk_major, ijk_list, w_list = self._flux_weights(mesh, diffusion, connectivity, dr, ijk, axis, tissue_index_map)
        major_to_center = nonzero_weights(mesh, ijk, ijk_list, w_list, tissue_index_map, direction=1)
        center_to_major = nonzero_weights(mesh, ijk_major, ijk_list, w_list, tissue_index_map, direction=-1)

        rows = np.concatenate([major_to_center[0], center_to_major[0]])
        cols = np.concatenate([major_to_center[1], center_to_major[1]])
        weights = np.concatenate([major_to_center[2], center_to_major[2]]) / dr

        return rows, cols, weights

    def _flux_weights(self, mesh, diffusion, connectivity, dr, ijk, major_axis, tissue_index_map):
        """Build face flux coefficients for the selected axis.

        Parameters
        ----------
        mesh : numpy.ndarray
            Grid labels: 0 for empty, 1 for active, 2 for fibrotic nodes.
            All mesh > 0 nodes are included in compact C-order matrix indexing.
        diffusion : float or numpy.ndarray
            Isotropic coefficient or tensors of shape (N_tissue, ndim, ndim),
            ordered by flat indices where mesh > 0. N_tissue counts nonempty nodes.
        connectivity : float or numpy.ndarray
            Connection factors: scalar, one per axis, or an array with shape
            (N_tissue, ndim), passed to the stencil connectivity helper.
        dr : float
            Uniform grid spacing along every axis.
        ijk : numpy.ndarray, shape (ndim, N_points)
            Full-grid coordinates of center nodes.
        major_axis : int
            Axis of the face-normal flux.
        tissue_index_map : numpy.ndarray, shape mesh.shape
            Map full-grid coordinates to compact nonempty-node indices.

        Returns
        -------
        ijk_major : numpy.ndarray, shape (ndim, N_points)
            Positive-axis neighbor coordinates.
        ijk_list : list of numpy.ndarray
            Coordinates used by each flux contribution.
        w_list : list of numpy.ndarray
            Flux coefficients containing one factor of 1/dr.
        """
        ijk_major = build_neighbor(ijk, shift=1, axis=major_axis)
        m_major = is_valid_index(ijk_major, mesh)

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
        """Build normal flux weights: -Daa * (u_major - u_center) / dr.

        Parameters
        ----------
        diffusion_major : numpy.ndarray, shape (N_points,)
            Averaged diagonal diffusion component including connection factors.
        dr : float
            Uniform grid spacing along every axis.
        ijk : numpy.ndarray, shape (ndim, N_points)
            Full-grid coordinates of center nodes.
        ijk_major : numpy.ndarray, shape (ndim, N_points)
            Full-grid coordinates of major neighbors.
        m_major : numpy.ndarray, shape (N_points,)
            Boolean mask of valid major neighbors.

        Returns
        -------
        ijk_list : list of numpy.ndarray
            Coordinates used by each flux contribution.
        w_list : list of numpy.ndarray
            Flux coefficients containing one factor of 1/dr.
        """
        w_major = np.where(m_major > 0, diffusion_major / dr, 0.)

        ijk_list = [ijk, ijk_major]
        w_list = [w_major, -w_major]
        return ijk_list, w_list

    def _minor_flux_weights(self, diffusion_minor, dr, ijk_center, ijk_major,
                           m_major, minor_axis, mesh):
        """Build transverse flux weights using four neighboring values.

        The coefficient is D_ab / (4 * dr). The contribution is zero unless
        the major neighbor and all four transverse neighbors are active.

        Parameters
        ----------
        diffusion_minor : numpy.ndarray, shape (N_points,)
            Averaged cross-diffusion component including connection factors.
        dr : float
            Uniform grid spacing along every axis.
        ijk_center : numpy.ndarray, shape (ndim, N_points)
            Full-grid coordinates of center nodes.
        ijk_major : numpy.ndarray, shape (ndim, N_points)
            Full-grid coordinates of major neighbors.
        m_major : numpy.ndarray, shape (N_points,)
            Boolean mask of valid major neighbors.
        minor_axis : int
            Transverse grid axis.
        mesh : numpy.ndarray
            Grid labels: 0 for empty, 1 for active, 2 for fibrotic nodes.
            All mesh > 0 nodes are included in compact C-order matrix indexing.

        Returns
        -------
        ijk_list : list of numpy.ndarray
            Coordinates used by each flux contribution.
        w_list : list of numpy.ndarray
            Flux coefficients containing one factor of 1/dr.
        """
        ijk_1 = build_neighbor(ijk_center, -1, minor_axis)
        ijk_2 = build_neighbor(ijk_major, -1, minor_axis)
        ijk_3 = build_neighbor(ijk_center, 1, minor_axis)
        ijk_4 = build_neighbor(ijk_major, 1, minor_axis)

        m1 = is_valid_index(ijk_1, mesh)
        m2 = is_valid_index(ijk_2, mesh)
        m3 = is_valid_index(ijk_3, mesh)
        m4 = is_valid_index(ijk_4, mesh)

        mask = (m_major > 0) & (m1 > 0) & (m2 > 0) & (m3 > 0) & (m4 > 0)

        w = np.where(mask, diffusion_minor / (4 * dr), 0.)
        ijk_list = [ijk_1, ijk_2, ijk_3, ijk_4]
        w_list = [w, w, -w, -w]
        return ijk_list, w_list

    def _diffusion_tensor_component(self, diffusion, connectivity, ijk, ijk_major,
                                   mask_major, major_axis, tissue_index_map):
        """Combine face-averaged diffusion with connection factors.

        Parameters
        ----------
        diffusion : float or numpy.ndarray
            Isotropic coefficient or tensors of shape (N_tissue, ndim, ndim),
            ordered by flat indices where mesh > 0. N_tissue counts nonempty nodes.
        connectivity : float or numpy.ndarray
            Connection factors: scalar, one per axis, or an array with shape
            (N_tissue, ndim), passed to the stencil connectivity helper.
        ijk : numpy.ndarray, shape (ndim, N_points)
            Full-grid coordinates of center nodes.
        ijk_major : numpy.ndarray, shape (ndim, N_points)
            Full-grid coordinates of major neighbors.
        mask_major : numpy.ndarray, shape (N_points,)
            Boolean mask of valid major neighbors.
        major_axis : int
            Axis of the face-normal flux.
        tissue_index_map : numpy.ndarray, shape mesh.shape
            Map full-grid coordinates to compact nonempty-node indices.

        Returns
        -------
        component : numpy.ndarray, shape (N_points, ndim)
            Tensor row along the major axis, scaled by connectivity.
        """
        diffusion_along_major = self._average_diffusion_component(
            diffusion, ijk, ijk_major, mask_major, major_axis, tissue_index_map)
        connectivity_along_major = self._connectivity_component(
            connectivity, ijk, mask_major, major_axis, tissue_index_map)
        return diffusion_along_major * connectivity_along_major[:, None]

    def _average_diffusion_component(self, diffusion, ijk, ijk_neighbor, mask, axis, tissue_index_map):
        """Average a tensor row between center and valid neighbor nodes.

        Parameters
        ----------
        diffusion : float or numpy.ndarray
            Isotropic coefficient or tensors of shape (N_tissue, ndim, ndim),
            ordered by flat indices where mesh > 0. N_tissue counts nonempty nodes.
        ijk : numpy.ndarray, shape (ndim, N_points)
            Full-grid coordinates of center nodes.
        ijk_neighbor : numpy.ndarray, shape (ndim, N_points)
            Full-grid coordinates of candidate neighbors.
        mask : numpy.ndarray, shape (N_points,)
            Boolean mask of valid neighbors.
        axis : int
            Grid axis.
        tissue_index_map : numpy.ndarray, shape mesh.shape
            Map full-grid coordinates to compact nonempty-node indices.

        Returns
        -------
        component : numpy.ndarray, shape (N_points, ndim)
            Averaged tensor rows. Scalar input produces the corresponding row
            of diffusion times identity; invalid-face weights are masked later.
        """
        diffusion = np.atleast_1d(diffusion)
        ndim, n_points = ijk_neighbor.shape
        if diffusion.size == 1:
            return np.broadcast_to(diffusion * np.eye(ndim)[axis], (n_points, ndim))

        center_indexes = tissue_index_map[*ijk[:, mask > 0]]
        neighbor_indexes = tissue_index_map[*ijk_neighbor[:, mask > 0]]
        d_axis = self.diffusion_averaging_method(
            diffusion[center_indexes, axis, :], diffusion[neighbor_indexes, axis, :])

        d_full = np.zeros((n_points, ndim))
        d_full[mask > 0] = d_axis
        return d_full

    def _connectivity_component(self, connectivity, ijk, mask, axis, tissue_index_map):
        """Select connection factors along one grid axis.

        Scalar and axis-wise inputs return a length-one array for broadcasting.
        For array input, the implementation copies connectivity[:, axis] and
        sets entries at valid center indices to zero.

        Parameters
        ----------
        connectivity : float or numpy.ndarray
            Connection factors: scalar, one per axis, or an array with shape
            (N_tissue, ndim), passed to the stencil connectivity helper.
        ijk : numpy.ndarray, shape (ndim, N_points)
            Full-grid coordinates of center nodes.
        mask : numpy.ndarray, shape (N_points,)
            Boolean mask of valid neighbors.
        axis : int
            Grid axis.
        tissue_index_map : numpy.ndarray, shape mesh.shape
            Map full-grid coordinates to compact nonempty-node indices.

        Returns
        -------
        component : numpy.ndarray
            Selected connection factors.
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
