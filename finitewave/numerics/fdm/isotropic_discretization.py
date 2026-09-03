import numpy as np
from .asymmetric_discretization import AsymmetricDiscretization


class IsotropicDiscretization(AsymmetricDiscretization):
    """
    Isotropic finite difference discretization with second-order accuracy for boundary.
    """
    
    def _diffusion_operator_component(self, mesh, diffusion, connectivity, dr, ijk, axis, tissue_index_map):
        """
        Computes the diffusion weights from the flux weights.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        diffusion : numpy.ndarray [*mesh.shape, ndim, ndim]
            The diffusion tensor at connections between nodes.
        dr : float
            The grid spacing.
        ijk : numpy.ndarray
            The indexes of the non-empty nodes in the mesh.
        axis : int
            The axis along which to compute the diffusion weights.

        Returns
        -------
        rows : np.ndarray
            The row indexes for the sparse matrix.
        cols : np.ndarray
            The column indexes for the sparse matrix.
        weights : np.ndarray
            The weights for the sparse matrix.
        """
        ijk_list, w_list = self._flux_weights(mesh, diffusion, connectivity, dr, ijk, axis, tissue_index_map)
        rows, cols, weights = self.nonzero_weights(mesh, ijk, ijk_list, w_list, tissue_index_map, direction=1)

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights) / dr
        return rows, cols, weights

    def _flux_weights(self, mesh, diffusion, connectivity, dr, ijk, axis, tissue_index_map):
        """
        Computes the flux weights along a given axis using the formula:

        q = - D * (u_neighbor - u_center) / dr

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
        axis : int
            The axis along which to compute the flux weights.

        Returns
        -------
        ijk_list : list
            The list of coordinates of the involved nodes in the mesh.
        w_list : list
            The list of weights for the involved nodes.
        """
        ijk_pos = self.build_neighbor(ijk, shift=1, axis=axis)
        ijk_neg = self.build_neighbor(ijk, shift=-1, axis=axis)

        valid_pos = self.is_valid_index(ijk_pos, mesh)
        valid_neg = self.is_valid_index(ijk_neg, mesh)

        d_pos = self.diffusion_tensor_component(diffusion, connectivity, ijk, ijk_pos, valid_pos, axis, tissue_index_map)
        d_neg = self.diffusion_tensor_component(diffusion, connectivity, ijk, ijk_neg, valid_neg, axis, tissue_index_map)

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
