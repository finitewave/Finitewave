import numpy as np

from .stencil import Stencil


class SymmetricStencil(Stencil):
    """Implements a cell based finite difference stencil for
    computing diffusion weights in cardiac simulations.
    The stencil is designed to handle non-uniform diffusion tensors.
   
    References
    ----------
    Saleheen, H. I., & Ng, K. T. (2002).
    New finite difference formulations for general inhomogeneous anisotropic
    bioelectric problems.
    IEEE transactions on biomedical engineering, 44(9), 800-809.
    https://doi.org/10.1109/10.623049
    """
    def __init__(self):
        super().__init__()

    def compute_diffusion_weights(self, mesh, diffusion, dr, indexes):
        """Computes the weights for the symmetric stencil.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        diffusion : numpy.ndarray [*mesh.shape, ndim, ndim]
            The diffusion tensor at cell centers.
        dr : float
            The grid spacing.
        indexes : numpy.ndarray
            The indexes of the non-empty cells in the mesh.

        Returns
        -------
        rows : numpy.ndarray
            The row indices for the sparse matrix.
        cols : numpy.ndarray
            The column indices for the sparse matrix.
        weights : numpy.ndarray
            The weights for the sparse matrix.
        """
        cond = mesh.copy()
        cond[-1, :] = 0
        cond[:, -1] = 0
        indexes = np.flatnonzero(cond == 1)
        ijk = np.array(np.unravel_index(indexes, mesh.shape))
        ijk_c, ijk_list, w_list = self.compute_cell_weights(mesh, diffusion, dr, ijk)
        rows, cols, weights = self.nonzero_weights(mesh, ijk_c, ijk_list, w_list)

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights) / dr
        return rows, cols, weights

    def compute_cell_weights(self, mesh, diffusion, dr, ijk):
        """
        Computes the weights for the symmetric stencil based on the
        diffusion tensor at cell centers.

        For each cell, the stencil calculates the flux to its nodes:

            001 ------- 011
           / |         / |
          /  |        /  |
        101 ------- 111  |
         |   |       |   |
         |  000 -----|- 010
         | /         |  /
         |/          | /
        100 ------- 110

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        diffusion : numpy.ndarray [*mesh.shape, ndim, ndim]
            The diffusion tensor at cell centers.
        dr : float
            The grid spacing.
        ijk : numpy.ndarray
            The indices of the cell centers.

        Returns
        -------
        ijk_center : numpy.ndarray
            The indices of the cell centers.
        ijk_list : list of numpy.ndarray
            The indices of the neighboring nodes.
        w_list : list of numpy.ndarray
            The weights for the neighboring nodes.
        """
        pqr = np.meshgrid(*([np.arange(2)] * mesh.ndim), indexing='ij')
        pqr = np.stack([x.flatten() for x in pqr], axis=1)

        pqr_inv = (pqr == 0).astype(pqr.dtype)

        ijk_center = ijk[None, :, :] + pqr[:, :, None]
        ijk_center = np.concatenate(ijk_center, axis=1)

        mask_center = mesh[*ijk_center] != 1

        ijk_list = []
        w_list = []

        for i in range(mesh.ndim):
            pqr_neighbor = pqr.copy()
            pqr_neighbor[:, i] = pqr_inv[:, i]

            for j in range(i, mesh.ndim):
                
                pqr_neighbor[:, j] = pqr_inv[:, j]
                ijk_neighbor = ijk[None, :, :] + pqr_neighbor[:, :, None]
                ijk_neighbor = np.concatenate(ijk_neighbor, axis=1)

                d_ij = diffusion[*ijk, i, j]

                if i == j:
                    sign = np.ones(len(pqr), dtype=d_ij.dtype)
                else:
                    sign = 2 * (pqr[:, i] == pqr[:, j]).astype(d_ij.dtype) - 1.

                w = sign[:, None] * d_ij[None, :] / (dr * 2**(mesh.ndim - 1))
                w = w.reshape(-1)

                mask_neighbor = mesh[*ijk_neighbor] != 1

                w[mask_center | mask_neighbor] = 0.
                
                ijk_n = np.concatenate([ijk_center, ijk_neighbor], axis=1)
                w_n = np.concatenate([w.reshape(-1), -w.reshape(-1)])

                ijk_list.append(ijk_n)
                w_list.append(w_n)

        ijk_center = np.concatenate([ijk_center, ijk_center], axis=1)

        return ijk_center, ijk_list, w_list