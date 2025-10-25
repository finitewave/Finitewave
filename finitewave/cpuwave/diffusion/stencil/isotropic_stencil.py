import numpy as np
from scipy import sparse as sp
from .stencil import Stencil


class IsotropicStencil(Stencil):
    """
    This class computes the weights for diffusion on a 2D and 3D grid
    using an isotropic stencil. The stencil includes 4 neighbors in 2D and 6
    neighbors in 3D.

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
        rows, cols, weights = self.boundary.compute_weights(mesh, indexes)
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
        K_new = - (K + D)
        M = sp.diags(np.ones_like(row_sums), offsets=0, format='csr')
        return K_new.tocsr(), M.tocsr()


class IsotropicFirstOrderBoundary:
    def __init__(self):
        pass

    def compute_weights(self, mesh, indexes):
        ijk = np.array(np.unravel_index(indexes, mesh.shape))
        rows = []
        cols = []
        weights = []

        for i_axis in range(len(ijk)):
            neighbors_left = ijk.copy()
            neighbors_right = ijk.copy()
            neighbors_left[i_axis] -= 1
            neighbors_right[i_axis] += 1

            left = self.is_valid_neighbor(neighbors_left, mesh, i_axis)
            right = self.is_valid_neighbor(neighbors_right, mesh, i_axis)
            left, right = self.apply_boundary_rule(left, right)

            rows.append(indexes[left > 0])
            cols.append(np.ravel_multi_index(neighbors_left[:, left > 0],
                                             mesh.shape))
            weights.append(left[left > 0])

            rows.append(indexes[right > 0])
            cols.append(np.ravel_multi_index(neighbors_right[:, right > 0],
                                             mesh.shape))
            weights.append(right[right > 0])

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights)
        return rows, cols, weights

    def is_valid_neighbor(self, neighbors, mesh, i_axis):
        mask = ((neighbors[i_axis] >= 0) &
                (neighbors[i_axis] < mesh.shape[i_axis]))
        mask[mask] = mesh[tuple(neighbors[:, mask])] > 0
        return mask.astype(np.int64)

    def apply_boundary_rule(self, left, right):
        return left, right


class IsotropicSecondOrderBoundary(IsotropicFirstOrderBoundary):
    def __init__(self):
        super().__init__()

    def apply_boundary_rule(self, left, right):
        left_empty = (left == 0).astype(np.int64)
        right_empty = (right == 0).astype(np.int64)

        left *= (left + right_empty)
        right *= (right + left_empty)
        return left, right


# class IsotropicSecondOrderOneSided:
#     def __init__(self):
#         super().__init__()

#     def find_neighbors(self, mesh, indexes):
#         ijk = np.array(np.unravel_index(indexes, mesh.shape))
#         rows = np.empty((len(ijk), 2, len(indexes)), dtype=np.int64)
#         cols = np.empty((len(ijk), 2, len(indexes)), dtype=np.int64)
#         weights = np.zeros((len(ijk), 2, len(indexes)), dtype=np.int64)

#         for i_axis in range(len(ijk)):
#             for i, neighbor in enumerate([-1, 1]):
#                 neighbors = ijk.copy()
#                 neighbors[i_axis] += neighbor
#                 rows[i_axis, i, :] = indexes
#                 cols[i_axis, i, :] = np.ravel_multi_index(neighbors,
#                                                           mesh.shape)
#                 weights[i_axis, i, :] = (
#                     (neighbors[i_axis] < 0) |
#                     (neighbors[i_axis] >= mesh.shape[i_axis]) |
#                     (mesh.flat[cols[i_axis, i, :]] != 1)).astype(np.int64)
#         return rows, cols, weights

#     def compute(self, mesh, indexes):
#         rows, cols, weights = self.find_neighbors(mesh, indexes)
#         return rows[weights], cols[weights]

