import numpy as np
from .stecil import Stencil


class IsotropicStencil(Stencil):
    """
    Assembles the isotropic stencil with first-order boundary conditions.
    """
    def __init__(self):
        super().__init__()

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
        d_axis = diffusion[..., axis, axis]
        ijk_neighbor = self.build_neighbor(ijk, shift=1, axis=axis)
        valid_connection = (self.is_valid_index(ijk_neighbor, mesh) &
                            self.is_valid_index(ijk, mesh))
        weights = - self.average_diffusion(d_axis, ijk, ijk_neighbor, valid_connection) / dr
        ijk_list = [ijk, ijk_neighbor]
        w_list = [-weights, weights]
        return ijk_neighbor, ijk_list, w_list

    def average_diffusion(self, diffusion, ijk, ijk_neighbor, mask):
        """
        Computes the average diffusion between the center and neighbor cells.

        Parameters
        ----------
        diffusion : np.ndarray
            The diffusion coefficients.
        ijk : np.ndarray
            The indices of the center points.
        ijk_neighbor : np.ndarray
            The indices of the neighbor points.
        mask : np.ndarray
            The mask indicating valid neighbors.

        Returns
        -------
        np.ndarray
            The average diffusion coefficients.
        """
        d_avg = np.zeros(mask.shape, dtype=diffusion.dtype)
        d_avg[mask > 0] = 0.5 * (diffusion[tuple(ijk[:, mask > 0])] +
                                 diffusion[tuple(ijk_neighbor[:, mask > 0])])
        return d_avg
