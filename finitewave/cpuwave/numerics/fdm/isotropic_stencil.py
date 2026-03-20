import numpy as np
from .stencil import Stencil


class IsotropicStencil(Stencil):
    """
    Isotropic stencil with second-order accuracy
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
        tuple of np.ndarray
            The rows, columns, and weights for the sparse matrix.
        """
        rows = []
        cols = []
        weights = []

        ijk = np.array(np.unravel_index(indexes, mesh.shape))
        for axis in range(mesh.ndim):
            res = self.compute_flux_weights(mesh, diffusion, dr, ijk, axis)
            r, c, w = self.compute_diffusion_component(mesh, dr, ijk, *res)
            rows.append(r)
            cols.append(c)
            weights.append(w)

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights)
        return rows, cols, weights
    
    def compute_diffusion_component(self, mesh, dr, ijk, ijk_list, w_list):
        """
        Computes the diffusion weights from the flux weights.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        dr : float
            The grid spacing.
        ijk : numpy.ndarray
            The indexes of the non-empty points in the mesh.
        ijk_list : list
            The list of ijk coordinates of the involved points in the mesh.
        w_list : list
            The list of weights for the involved points.

        Returns
        -------
        rows : np.ndarray
            The row indexes for the sparse matrix.
        cols : np.ndarray
            The column indexes for the sparse matrix.
        weights : np.ndarray
            The weights for the sparse matrix.
        """
        rows, cols, weights = self.nonzero_weights(mesh, ijk, ijk_list, w_list)

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights) / dr
        return rows, cols, weights

    def compute_flux_weights(self, mesh, diffusion, dr, ijk, axis):
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
        tuple of np.ndarray
            The rows, columns, and weights for the flux computation.
        """
        ijk_pos = self.build_neighbor(ijk, shift=1, axis=axis)
        ijk_neg = self.build_neighbor(ijk, shift=-1, axis=axis)

        valid_pos = self.is_valid_index(ijk_pos, mesh)
        valid_neg = self.is_valid_index(ijk_neg, mesh)

        d_pos = np.zeros(valid_pos.shape, dtype=diffusion.dtype)
        d_neg = np.zeros(valid_neg.shape, dtype=diffusion.dtype)

        d_pos[valid_pos] = diffusion[*ijk[:, valid_pos], axis, axis]
        d_neg[valid_neg] = diffusion[*ijk_neg[:, valid_neg], axis, axis]

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
