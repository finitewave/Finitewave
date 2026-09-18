import numpy as np
from scipy import sparse
from ._grid_utils import nonzero_weights, build_neighbor, is_valid_index


class FiniteDifferenceGradient:
    """Build grid gradients with centered and one-sided differences.

    Matrices use compact C-order indexing of all nodes with ``mesh > 0``.
    Neighbors must have ``mesh == 1``; isolated nodes have zero gradients.
    """

    def build_gradient_operator(self, mesh, *, dr=1.0, indexes=None, **kwargs):
        """Build centered or one-sided grid gradient operators.

        Parameters
        ----------
        mesh : numpy.ndarray
            Grid labels: 0 for empty, 1 for active, 2 for fibrotic nodes.
            All mesh > 0 nodes are included in compact C-order matrix indexing.
        dr : float, optional
            Uniform grid spacing along every axis. Default is 1.0.
        indexes : numpy.ndarray, optional
            Flat full-grid indices of center nodes. For operator builders,
            None selects mesh == 1; these are not compact matrix indices. Default is None.
        **kwargs : dict
            Ignored; accepted for compatibility with other discretizations.

        Returns
        -------
        grad_ops : list of scipy.sparse.csr_matrix
            One matrix per grid axis, each of shape (N_tissue, N_tissue).
            Apply to values ordered by flat indices where mesh > 0. Uses centered
            differences with two active neighbors, one-sided differences with one,
            and zero with neither. Rows outside indexes remain zero.
        """
        if indexes is None:
            indexes = np.flatnonzero(mesh == 1)

        tissue_size = np.count_nonzero(mesh > 0)
        tissue_index_map = - np.ones_like(mesh, dtype=np.int64)
        tissue_index_map[mesh > 0] = np.arange(tissue_size)

        if np.any(tissue_index_map.flat[indexes] < 0):
            raise ValueError("Tissue index mapping failed. Check the mesh and indexes.")

        ijk = np.array(np.unravel_index(indexes, mesh.shape))

        grad_ops = []
        for axis in range(mesh.ndim):

            ijk_pos = build_neighbor(ijk, 1, axis)
            ijk_neg = build_neighbor(ijk, -1, axis)

            is_valid_pos = is_valid_index(ijk_pos, mesh)
            is_valid_neg = is_valid_index(ijk_neg, mesh)

            is_valid = is_valid_pos | is_valid_neg
            invalid_pos = (~is_valid_pos) & is_valid_neg
            invalid_neg = (~is_valid_neg) & is_valid_pos
            one_sided = invalid_pos | invalid_neg

            ijk_pos[:, invalid_pos] = ijk[:, invalid_pos]
            ijk_neg[:, invalid_neg] = ijk[:, invalid_neg]

            w_pos = np.zeros(len(indexes), dtype=np.float64)
            w_neg = np.zeros(len(indexes), dtype=np.float64)

            w_pos[is_valid] = 1.0 / (2 * dr)
            w_neg[is_valid] = -1.0 / (2 * dr)

            w_pos[one_sided] = 1.0 / dr
            w_neg[one_sided] = -1.0 / dr

            ijk_list = [ijk_pos, ijk_neg]
            w_list = [w_pos, w_neg]

            rows, cols, weights = nonzero_weights(mesh, ijk, ijk_list,
                                                        w_list, tissue_index_map,
                                                        direction=1)
            grad_ops.append(
                sparse.coo_matrix((weights, (rows, cols)),
                                  shape=(tissue_size, tissue_size)).tocsr()
            )
        return grad_ops
