import numpy as np
from scipy import sparse
from .finite_difference_discretization import FiniteDifferenceDiscretization


class AsymmetricDiscretization(FiniteDifferenceDiscretization):
    """Conservative finite differences for tensor-valued diffusion.

    The method constructs the flux through every positive grid face and adds
    that flux with opposite signs to the two adjacent rows.  Diagonal tensor
    terms use the two face-adjacent cells; off-diagonal terms use four cells
    surrounding the face.  The assembled CSR matrix approximates
    ``-div(D grad(u))`` and conserves flux: every face expression is added to
    one row and subtracted from the adjacent row.

    Notes
    -----
    With scalar diffusion, or an axis-aligned diagonal tensor, the interior
    stencil reduces to the usual centered stencil.  Its boundary treatment is
    different from :class:`IsotropicDiscretization`: a missing face has zero
    flux instead of a mirrored opposite contribution.

    Despite flux conservation, the matrix is generally non-symmetric when
    off-diagonal tensor components are present, even if the continuous tensor
    is symmetric.  ``Asymmetric`` in the class name refers to this stencil
    property.  A solver that requires a symmetric positive-definite matrix,
    such as Conjugate Gradient, is therefore not guaranteed to be applicable
    to the fully anisotropic operator.

    Boundary rules are as follows:

    * If the major (directly adjacent) neighbor is invalid, the complete flux
      through that face is zero.
    * If any of the four points required for a transverse derivative is
      invalid, that transverse contribution is zero.  The normal contribution
      may still remain.

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
        """Select how diffusion tensors are interpolated to grid faces.
        
        Parameters
        ----------
        averaging_method : str, optional
            ``"arithmetic"`` averages tensor-row components elementwise.
            ``"harmonic"`` uses ``2*d1*d2/(d1+d2)`` elementwise and returns
            zero where the denominator is zero.

        Raises
        ------
        ValueError
            If ``averaging_method`` is not ``"arithmetic"`` or ``"harmonic"``.
        """
        super().__init__()
        if averaging_method == "arithmetic":
            self.diffusion_averaging_method = self._arithmetic_mean
        elif averaging_method == "harmonic":
            self.diffusion_averaging_method = self._harmonic_mean
        else:
            raise ValueError(f"Invalid averaging method: {averaging_method}. "
                             "Choose 'arithmetic' or 'harmonic'.")

    def compute_diffusion_operator(self, mesh, dr, indexes=None, diffusion=1., connectivity=1.):
        """Assemble ``K``, the sparse approximation of ``-div(D grad(u))``.

        Matrix rows and columns use compressed tissue indexing based on
        ``mesh > 0``.  Only coordinates selected by ``indexes`` generate flux
        faces; the returned matrix nevertheless includes every tissue point.

        Parameters
        ----------
        mesh : numpy.ndarray
            Integer grid where ``0`` is empty, ``1`` is active, and ``2`` is a
            retained non-excitable point.
        dr : float
            The grid spacing.
        indexes : numpy.ndarray, optional
            Positions of active cells in the compressed tissue array
            ``mesh[mesh > 0]``. By default all cells where ``mesh == 1``.
        diffusion : scalar or numpy.ndarray
            A scalar isotropic coefficient, a constant ``(ndim, ndim)``
            tensor, a full-grid field with shape
            ``mesh.shape + (ndim, ndim)``, or a compressed field with shape
            ``(n_tissue, ndim, ndim)``.
        connectivity : scalar or numpy.ndarray
            Positive-edge multiplier. ``connectivity[p, a]`` scales the face
            joining point ``p`` to its ``+a`` neighbor.  Accepted storage is a
            scalar, an ``(ndim,)`` vector, ``mesh.shape + (ndim,)``, or
            ``(n_tissue, ndim)``.

        Returns
        -------
        scipy.sparse.csr_matrix
            Square matrix with shape ``(count(mesh > 0),) * 2``.  Duplicate
            face contributions are summed during CSR construction.  Rows sum
            to zero, but the matrix need not be symmetric for tensor-valued
            diffusion.

        Raises
        ------
        ValueError
            If ``dr`` is non-positive or an input shape/index contract is
            violated.
        """
        rows = []
        cols = []
        weights = []

        if dr <= 0:
            raise ValueError("dr must be positive.")

        tissue_flat_indexes = np.flatnonzero(mesh > 0)
        tissue_size = tissue_flat_indexes.size

        if indexes is None:
            indexes = np.flatnonzero(mesh.flat[tissue_flat_indexes] == 1)
        else:
            indexes = np.asarray(indexes, dtype=np.int64)

        if indexes.ndim != 1:
            raise ValueError("indexes must be a one-dimensional array.")

        if np.any((indexes < 0) | (indexes >= tissue_size)):
            raise ValueError("indexes must refer to positions in the tissue array.")

        myo_flat_indexes = tissue_flat_indexes[indexes]
        if np.any(mesh.flat[myo_flat_indexes] != 1):
            raise ValueError("indexes must only refer to active cells (mesh == 1).")

        diffusion = self._normalize_diffusion(diffusion, mesh, tissue_flat_indexes)
        connectivity = self._normalize_connectivity(
            connectivity, mesh, tissue_flat_indexes
        )

        tissue_index_map = - np.ones(mesh.shape, dtype=indexes.dtype)
        tissue_index_map[mesh > 0] = np.arange(tissue_size, dtype=indexes.dtype)

        ijk = np.array(np.unravel_index(myo_flat_indexes, mesh.shape))

        for axis in range(mesh.ndim):
            r, c, w = self._diffusion_operator_component(mesh, diffusion, connectivity, dr, ijk, axis, tissue_index_map)
            rows.append(r)
            cols.append(c)
            weights.append(w)

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights)
        return sparse.csr_matrix((weights, (rows, cols)), shape=(tissue_size, tissue_size))

    @staticmethod
    def _arithmetic_mean(d1, d2):
        """Return the elementwise arithmetic mean of two face values."""
        return 0.5 * (d1 + d2)

    @staticmethod
    def _harmonic_mean(d1, d2):
        """Return the elementwise harmonic mean, using zero for a zero sum."""
        denominator = d1 + d2
        result = np.zeros_like(denominator, dtype=np.result_type(d1, d2, float))
        return np.divide(2 * d1 * d2, denominator, out=result,
                         where=denominator != 0)

    @staticmethod
    def _normalize_diffusion(diffusion, mesh, tissue_flat_indexes):
        """Normalize supported diffusion layouts to scalar or tissue order.

        A constant ``(ndim, ndim)`` tensor and a full-grid tensor field are
        converted to ``(n_tissue, ndim, ndim)``.  A scalar is kept scalar so
        downstream code can use its isotropic fast path.
        """
        diffusion = np.asarray(diffusion)
        ndim = mesh.ndim
        tissue_size = tissue_flat_indexes.size

        if diffusion.size == 1:
            return diffusion
        if diffusion.shape == (ndim, ndim):
            return np.broadcast_to(diffusion, (tissue_size, ndim, ndim))
        if diffusion.shape == mesh.shape + (ndim, ndim):
            return diffusion.reshape((-1, ndim, ndim))[tissue_flat_indexes]
        if diffusion.shape == (tissue_size, ndim, ndim):
            return diffusion

        raise ValueError(
            "diffusion must be scalar, a constant (ndim, ndim) tensor, "
            "a full-grid tensor, or a tissue-indexed tensor."
        )

    @staticmethod
    def _normalize_connectivity(connectivity, mesh, tissue_flat_indexes):
        """Normalize connectivity to scalar, axis-wise, or tissue order.

        Full-grid data are compressed using ``tissue_flat_indexes``.  Scalars
        and constant ``(ndim,)`` vectors are kept in their compact form.
        """
        connectivity = np.asarray(connectivity)
        ndim = mesh.ndim
        tissue_size = tissue_flat_indexes.size

        if connectivity.size == 1 or connectivity.shape == (ndim,):
            return connectivity
        if connectivity.shape == mesh.shape + (ndim,):
            return connectivity.reshape((-1, ndim))[tissue_flat_indexes]
        if connectivity.shape == (tissue_size, ndim):
            return connectivity

        raise ValueError(
            "connectivity must be scalar, an (ndim,) vector, a full-grid "
            "array, or a tissue-indexed array."
        )
    
    def _diffusion_operator_component(self, mesh, diffusion, connectivity, dr, ijk, axis, tissue_index_map):
        """Convert positive-face fluxes for one axis into COO triplets.

        For each face, the same flux expression is added to the center row and
        subtracted from the positive-neighbor row.  Division by ``dr`` here is
        the discrete divergence; flux weights already contain another
        ``1 / dr`` from the discrete gradient.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        diffusion : scalar or numpy.ndarray
            Normalized scalar or tissue-indexed diffusion tensors.
        connectivity : scalar or numpy.ndarray
            Normalized positive-edge connectivity.
        dr : float
            The grid spacing.
        ijk : numpy.ndarray
            Active center coordinates with shape ``(mesh.ndim, n_active)``.
        axis : int
            Normal direction of the positive faces.
        tissue_index_map : numpy.ndarray
            Grid-shaped map to compressed tissue indexes.

        Returns
        -------
        rows : np.ndarray
            Compressed row indexes for both cells adjacent to valid faces.
        cols : np.ndarray
            Compressed indexes of cells used by the flux expressions.
        weights : np.ndarray
            COO values after both gradient and divergence scaling.
        """
        
        ijk_major, ijk_list, w_list = self._flux_weights(mesh, diffusion, connectivity, dr, ijk, axis, tissue_index_map)
        major_to_center = self.nonzero_weights(mesh, ijk, ijk_list, w_list, tissue_index_map, direction=1)
        center_to_major = self.nonzero_weights(mesh, ijk_major, ijk_list, w_list, tissue_index_map, direction=-1)

        rows = np.concatenate([major_to_center[0], center_to_major[0]])
        cols = np.concatenate([major_to_center[1], center_to_major[1]])
        weights = np.concatenate([major_to_center[2], center_to_major[2]]) / dr
        return rows, cols, weights
    
    def _flux_weights(self, mesh, diffusion, connectivity, dr, ijk, major_axis, tissue_index_map):
        """Build one linear flux expression for every positive-axis face.

        ``major_axis`` is the face normal.  The tensor row ``D[major_axis, :]``
        supplies one normal-gradient coefficient and, in anisotropic media,
        one transverse coefficient per minor axis.  Invalid major faces have
        no flux.  A transverse term is included only when all four surrounding
        minor points are active.

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
            Normalized scalar or tissue-indexed diffusion tensor.
        connectivity : scalar or numpy.ndarray
            Normalized multiplier of each positive face.
        dr : float
            The grid spacing.
        ijk : numpy.ndarray
            Active center coordinates with shape ``(mesh.ndim, n_active)``.
        major_axis : int
            The axis of the major direction.
        tissue_index_map : numpy.ndarray
            Grid-shaped map to compressed tissue indexes.

        Returns 
        -------
        ijk_major : numpy.ndarray
            Positive-neighbor coordinates, including invalid coordinates whose
            flux weights are zero.
        ijk_list : list
            Coordinate arrays used in the linear face-flux expressions.
        w_list : list
            Matching one-dimensional coefficient arrays.  These contain the
            gradient factor ``1 / dr`` but not the divergence factor.
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
        """Build the normal-gradient part of a positive-face flux.

        The returned coefficients represent
        ``D_aa * (u_center - u_major) / dr``.  Both coefficients are zero when
        the positive neighbor is invalid.
        
        Parameters
        ----------
        diffusion_major : numpy.ndarray
            The diffusion coefficient along the major axis.
        dr : float
            The grid spacing.
        ijk : numpy.ndarray
            Active center coordinates with shape ``(mesh.ndim, n_points)``.
        ijk_major : numpy.ndarray
            Positive-neighbor coordinates with the same shape as ``ijk``.
        m_major : numpy.ndarray
            Boolean vector selecting valid positive faces.

        Returns
        -------
        ijk_list : list of numpy.ndarray
            ``[ijk, ijk_major]``.
        w_list : list of numpy.ndarray
            Coefficients ``[w, -w]`` for center and major values.
        """
        w_major = np.where(m_major > 0, diffusion_major / dr, 0.)

        ijk_list = [ijk, ijk_major]
        w_list = [w_major, -w_major]
        return ijk_list, w_list

    def _minor_flux_weights(self, diffusion_minor, dr, ijk_center, ijk_major,
                           m_major, minor_axis, mesh):
        """Build one transverse-gradient part of a positive-face flux.

        For minor axis ``b`` and major axis ``a``, the approximation is
        ``D_ab * (u_1 + u_2 - u_3 - u_4) / (4*dr)`` using the four cells around
        the face.  It is suppressed unless the major neighbor and all four
        minor cells are active.

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
            Active center coordinates with shape ``(mesh.ndim, n_points)``.
        ijk_major : numpy.ndarray
            Positive-neighbor coordinates with the same shape.
        m_major : numpy.ndarray
            Boolean vector selecting valid major faces.
        minor_axis : int
            The axis of the minor direction.

        Returns
        -------
        ijk_list : list of numpy.ndarray
            Coordinates of the four cells surrounding each face.
        w_list : list of numpy.ndarray
            Coefficients ``[w, w, -w, -w]`` in matching order.

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
        """Return the connectivity-scaled tensor row on each positive face."""
        diffusion_along_major = self._average_diffusion_component(
            diffusion, ijk, ijk_major, mask_major, major_axis, tissue_index_map)
        connectivity_along_major = self._connectivity_component(
            connectivity, ijk, mask_major, major_axis, tissue_index_map)
        return diffusion_along_major * connectivity_along_major[:, None]

    def _average_diffusion_component(self, diffusion, ijk, ijk_neighbor, mask, axis, tissue_index_map):
        """Interpolate tensor row ``axis`` from cell centers to valid faces.

        The result always has shape ``(n_points, mesh.ndim)``.  Invalid faces
        are filled with zeros.  Scalar diffusion populates only the diagonal
        component, which represents isotropic ``D * I``.
        """
        diffusion = np.asarray(diffusion)
        ndim, n_points = ijk_neighbor.shape
        d_full = np.zeros(
            (n_points, ndim), dtype=np.result_type(diffusion.dtype, float)
        )

        if diffusion.size == 1:
            d_full[mask, axis] = diffusion.item()
            return d_full

        center_indexes = tissue_index_map[*ijk[:, mask]]
        neighbor_indexes = tissue_index_map[*ijk_neighbor[:, mask]]
        d_axis = self.diffusion_averaging_method(
            diffusion[center_indexes, axis, :], diffusion[neighbor_indexes, axis, :])

        d_full[mask] = d_axis
        return d_full

    def _connectivity_component(self, connectivity, ijk, mask, axis, tissue_index_map):
        """Read the multiplier stored at each center's positive-axis edge.

        Parameters
        ----------
        connectivity : scalar or numpy.ndarray
            Normalized scalar, ``(ndim,)`` vector, or tissue-indexed array.
        ijk : numpy.ndarray
            Center coordinates with shape ``(mesh.ndim, n_points)``.
        mask : numpy.ndarray
            Boolean vector selecting valid positive faces.
        axis : int
            The axis along which to compute the connectivity.

        Returns
        -------
        numpy.ndarray
            Vector of length ``n_points``; invalid faces contain zero.
        """
        connectivity = np.asarray(connectivity)
        ndim, n_points = ijk.shape
        connectivity_along_axis = np.zeros(
            n_points, dtype=np.result_type(connectivity.dtype, float)
        )

        if connectivity.size == 1:
            connectivity_along_axis[mask] = connectivity.item()
            return connectivity_along_axis

        if connectivity.shape == (ndim,):
            connectivity_along_axis[mask] = connectivity[axis]
            return connectivity_along_axis

        center_indexes = tissue_index_map[*ijk[:, mask]]
        connectivity_along_axis[mask] = connectivity[center_indexes, axis]
        return connectivity_along_axis
