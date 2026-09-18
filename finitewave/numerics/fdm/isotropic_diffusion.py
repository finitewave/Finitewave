import numpy as np
from .asymmetric_diffusion import AsymmetricDiffusion
from ._grid_utils import nonzero_weights, build_neighbor, is_valid_index


class IsotropicDiffusion(AsymmetricDiffusion):
    """Assemble diagonal diffusion with reflected Neumann boundaries.

    Parameters
    ----------
    averaging_method : {"arithmetic", "harmonic"}, optional
        Componentwise face averaging, default "arithmetic".

    Notes
    -----
    Uses only diagonal tensor entries; off-diagonal entries are ignored.
    If one neighbor is invalid, its coordinate and coefficient are replaced
    by those of the opposite valid neighbor. An isolated node contributes
    zero along that axis. Inherits compact indexing and matrix assembly
    from AsymmetricDiffusion.
    """

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
        ijk_list, w_list = self._flux_weights(mesh, diffusion, connectivity, dr, ijk, axis, tissue_index_map)
        rows, cols, weights = nonzero_weights(mesh, ijk, ijk_list, w_list, tissue_index_map, direction=1)

        weights = weights / dr
        return rows, cols, weights

    def _flux_weights(self, mesh, diffusion, connectivity, dr, ijk, axis, tissue_index_map):
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
        axis : int
            Grid axis.
        tissue_index_map : numpy.ndarray, shape mesh.shape
            Map full-grid coordinates to compact nonempty-node indices.

        Returns
        -------
        ijk_list : list of numpy.ndarray
            Coordinates used by each flux contribution.
        w_list : list of numpy.ndarray
            Flux coefficients containing one factor of 1/dr.
        """
        ijk_pos = build_neighbor(ijk, shift=1, axis=axis)
        ijk_neg = build_neighbor(ijk, shift=-1, axis=axis)

        valid_pos = is_valid_index(ijk_pos, mesh)
        valid_neg = is_valid_index(ijk_neg, mesh)

        d_pos = self._diffusion_tensor_component(
            diffusion, connectivity, ijk, ijk_pos, valid_pos, axis,
            tissue_index_map
            )[:, axis]
        d_neg = self._diffusion_tensor_component(
            diffusion, connectivity, ijk, ijk_neg, valid_neg, axis,
            tissue_index_map
            )[:, axis]

        d_pos = np.where(valid_pos, d_pos, 0.0)
        d_neg = np.where(valid_neg, d_neg, 0.0)

        invalid_pos = (~valid_pos) & valid_neg
        invalid_neg = (~valid_neg) & valid_pos

        ijk_pos[:, invalid_pos] = ijk_neg[:, invalid_pos]
        ijk_neg[:, invalid_neg] = ijk_pos[:, invalid_neg]

        d_pos[invalid_pos] = d_neg[invalid_pos]
        d_neg[invalid_neg] = d_pos[invalid_neg]

        d_pos /= dr
        d_neg /= dr

        ijk_list = [ijk, ijk_pos, ijk, ijk_neg]
        w_list = [d_pos, - d_pos, d_neg, - d_neg]

        return ijk_list, w_list
