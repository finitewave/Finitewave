import numpy as np
from .stecil import Stencil


class CellStencil(Stencil):
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
        pass

    def compute_diffusion_weights(self, mesh, diffusion, dr, indexes):
        """Computes the diffusion weights for the non-empty points in the mesh.
        
        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        diffusion : numpy.ndarray
            The diffusion tensor as a (*mesh.shape, ndim, ndim).
        dr : float
            The grid spacing.
        indexes : np.ndarray
            The indexes of the non-empty points in the mesh.
            
        Returns
        ---------
        rows : np.ndarray
            The row indexes for the sparse matrix.
        cols : np.ndarray
            The column indexes for the sparse matrix.
        weights : np.ndarray
            The weights for the sparse matrix.
        """
        if mesh.ndim == 2:
            scheme = StencilScheme2D()
        
        if mesh.ndim == 3:
            scheme = StencilScheme3D()
        
        ijk = np.array(np.unravel_index(indexes, mesh.shape))
        neighbors_ijk = self.build_all_neighbors(ijk, scheme.node_schemes)
        cells_ijk, cells_mask = self.build_all_cells(neighbors_ijk,
                                                     scheme.cell_schemes,
                                                     mesh, diffusion)
        w_list = self.build_weights(diffusion, dr, cells_ijk, cells_mask,
                                    scheme.weight_schemes)

        rows, cols, weights = self.nonzero_weights(mesh, ijk, neighbors_ijk, w_list)
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = - np.concatenate(weights)
        return rows, cols, weights
    
    def build_weights(self, diffusion, dr, cells_ijk, cells_mask, scheme):
        """Builds the diffusion weights for each node based on the fluxes of
        surrounding cells.
        
        Parameters
        ----------
        diffusion : numpy.ndarray [*mesh.shape, ndim, ndim]
            The diffusion tensor in the cells.
        dr : float
            The grid spacing.
        cells_ijk : list of np.ndarray [ndim, N_cells]
            The indexes of the cells for each node.
        cells_mask : list of np.ndarray [N_cells]
            A boolean array indicating whether each cell is valid
            (myocardial tissue).
        scheme : dict
            A dictionary containing the cell ids, axes, and signs to consider
            for each node.
            
        Returns
        -------
        list of np.ndarray
            The diffusion weights for each node.
        """
        w_list = []
        for cell_ids, axes, sign in zip(scheme['cell_ids'], scheme['axes'],
                                        scheme['signs']):
            w = self._compute_node_weights(diffusion, dr, cells_ijk, cells_mask,
                                           cell_ids, axes, sign)
            w_list.append(w)

        w_0 = - np.sum(w_list, axis=0)
        w_list = [w_0] + w_list
        return w_list
    
    def _compute_node_weights(self, diffusion, dr, cells_ijk, cells_mask,
                              cell_ids, axes, sign):
        """Computes the diffusion weights for a node based on the fluxes of
        surrounding cells.

        Parameters
        ----------
        diffusion : numpy.ndarray [*mesh.shape, ndim, ndim]
            The diffusion tensor in the cells.
        dr : float
            The grid spacing.
        cells_ijk : list of np.ndarray [ndim, N_cells]
            The indexes of the cells for each node.
        cells_mask : list of np.ndarray [N_cells]
            A boolean array indicating whether each cell is valid
            (myocardial tissue).
        cell_ids : list of int
            The indexes of the cells to consider for this node.
        axes : list of list of int
            The axes of the diffusion tensor to consider for each cell.
        sign : int
            The sign to apply to the weights from these cells.

        Returns
        -------
        np.ndarray
            The diffusion weights for the node.
        """
        w = []
        for cell_id, (axis_0, axis_1) in zip(cell_ids, axes):
            w.append(sign * self._node_diffusion(diffusion, dr,
                                                 cells_ijk[cell_id],
                                                 cells_mask[cell_id],
                                                 axis_0, axis_1))
        return np.sum(w, axis=0)
    
    def _node_diffusion(self, diffusion, dr, cell_indexes, cell_valid, axis_i,
                        axis_j):
        """Computes diffusion weights for a node which comes from the
        fluxes in the cell.
        
        Parameters
        ----------
        diffusion : numpy.ndarray [*mesh.shape, ndim, ndim]
            The diffusion tensor in the cells.
        dr : float
            The grid spacing.
        cell_indexes : numpy.ndarray [ndim, N_cells]
            The indexes of the cell.
        cell_valid : numpy.ndarray [N_cells]
            A boolean array indicating whether the cell is valid
            (myocardial tissue).
        axis_i : int
            The first axis of the diffusion tensor to consider.
        axis_j : int
            The second axis of the diffusion tensor to consider.

        Returns
        -------
        np.ndarray
            The diffusion weight for the node.
        """
        ndim = diffusion.shape[-1]
        return np.where(cell_valid, diffusion[*cell_indexes, axis_i, axis_j],
                        0.) / (2 * (ndim - 1) * dr**2)
    
    def build_all_neighbors(self, ijk, scheme):
        neighbors_ijk = []
        for shifts in scheme['shifts']:
            shifted_ijk = ijk.copy()
            for axis, shift in enumerate(shifts):
                shifted_ijk = self.build_neighbor(shifted_ijk, shift, axis)

            neighbors_ijk.append(shifted_ijk)

        return neighbors_ijk
    
    def build_all_cells(self, neighbors_ijk, scheme, mesh, diffusion):
        cells_ijk = [neighbors_ijk[i] for i in scheme['ids']]
        
        valid_cells = []
        for i, nodes in enumerate(scheme['nodes']):
            valid_cells.append(self.is_valid_cell(cells_ijk[i], neighbors_ijk,
                                                  nodes, mesh, diffusion))
        return cells_ijk, valid_cells
    
    def is_valid_cell(self, cell_ijk, ijk_list, cell_inds, mesh, diffusion):
        valid_cell = self.is_valid_index(ijk_list[cell_inds[0]], mesh)
        for point in cell_inds[1:]:
            valid_cell &= self.is_valid_index(ijk_list[point], mesh)

        # check if diffusion is non-zero in the cell
        diffusion_nonzero = np.any(diffusion[*cell_ijk] > 0., axis=(1, 2))
        return valid_cell & diffusion_nonzero


class StencilScheme2D:
    """Defines the node and cell schemes for a 2D cell-based finite
    difference stencil.

    Attributes
    ----------
    node_schemes : dict
        Defines the shifts to apply to the central node to get the neighboring
        nodes for the stencil.
    cell_schemes : dict
        Defines the nodes that make up each cell in the stencil and
        the corresponding ids for the cells
        (used for indexing the diffusion tensor).
    weight_schemes : dict
        Defines how to compute the weights for each node based on the fluxes
        in the surrounding cells, including which cells to consider, which axes
        of the diffusion tensor to use, and the signs to apply to the weights
        from each cell.
    
    References
    ----------
    Saleheen, H. I., & Ng, K. T. (2002).
    New finite difference formulations for general inhomogeneous anisotropic
    bioelectric problems.
    IEEE transactions on biomedical engineering, 44(9), 800-809.
    https://doi.org/10.1109/10.623049
    """
    def __init__(self):
        # shift_0, shift_1
        self.node_schemes = {
            'shifts': [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
                [1, 1],
                [-1, 1],
                [-1, -1],
                [1, -1]
            ]
        }
        
        self.cell_schemes = {
            'nodes':[
                [0, 1, 2, 5],
                [0, 2, 3, 6],
                [0, 3, 4, 7],
                [0, 1, 4, 8]
            ],
            'ids': [0, 3, 7, 4],
            }
        
        self.weight_schemes = {
            'cell_ids': [
                [0, 3],
                [0, 1],
                [1, 2],
                [2, 3],
                [0],
                [1],
                [2],
                [3]
            ],
            'axes': [
                [[0, 0], [0, 0]],
                [[1, 1], [1, 1]],
                [[0, 0], [0, 0]],
                [[1, 1], [1, 1]],
                [[0, 1]],
                [[0, 1]],
                [[0, 1]],
                [[0, 1]]
            ],
            'signs': [1, 1, 1, 1, 1, -1, 1, -1]
        }


class StencilScheme3D:
    """Defines the node and cell schemes for a 2D cell-based finite
    difference stencil.

    Attributes
    ----------
    node_schemes : dict
        Defines the shifts to apply to the central node to get the neighboring
        nodes for the stencil.
    cell_schemes : dict
        Defines the nodes that make up each cell in the stencil and
        the corresponding ids for the cells
        (used for indexing the diffusion tensor).
    weight_schemes : dict
        Defines how to compute the weights for each node based on the fluxes
        in the surrounding cells, including which cells to consider, which axes
        of the diffusion tensor to use, and the signs to apply to the weights
        from each cell.
    
    References
    ----------
    Saleheen, H. I., & Ng, K. T. (2002).
    New finite difference formulations for general inhomogeneous anisotropic
    bioelectric problems.
    IEEE transactions on biomedical engineering, 44(9), 800-809.
    https://doi.org/10.1109/10.623049
    """
    def __init__(self):
        # shift_0, shift_1, shift_2
        self.node_schemes = {
            'shifts': [
                [0, 0, 0],    # 0
                [1, 0, 0],    # 1
                [0, 1, 0],    # 2
                [-1, 0, 0],   # 3
                [0, -1, 0],   # 4
                [1, 1, 0],    # 5
                [-1, 1, 0],   # 6
                [-1, -1, 0],  # 7
                [1, -1, 0],   # 8
                [0, 0, 1],    # 9
                [0, 0, -1],   # 10
                [0, 1, 1],    # 11
                [0, 1, -1],   # 12
                [0, -1, -1],  # 13
                [0, -1, 1],   # 14
                [1, 0, 1],    # 15
                [-1, 0, 1],   # 16
                [-1, 0, -1],  # 17
                [1, 0, -1],   # 18
                [1, -1, -1],  # 19
                [1, 1, -1],   # 20
                [-1, 1, -1],  # 21
                [-1, -1, -1], # 22
                [1, -1, 1],   # 23
                [1, 1, 1],    # 24
                [-1, 1, 1],   # 25
                [-1, -1, 1]   # 26
            ]
        }

        self.cell_schemes = {
            'nodes': [
                [0, 2, 3, 6, 10, 12, 17, 21],
                [0, 3, 4, 7, 10, 13, 17, 22],
                [0, 1, 4, 8, 10, 13, 18, 19],
                [0, 1, 2, 5, 10, 12, 18, 20],
                [0, 2, 3, 6, 9, 11, 16, 25],
                [0, 3, 4, 7, 9, 14, 16, 26],
                [0, 1, 4, 8, 9, 14, 15, 23],
                [0, 1, 2, 5, 9, 11, 15, 24]
            ],
            'ids': [17, 22, 13, 10, 3, 7, 4, 0]
        }

        self.weight_schemes = {
            'cell_ids': [
                [2, 3, 6, 7],   # 1
                [0, 3, 4, 7],   # 2
                [0, 1, 4, 5],   # 3
                [1, 2, 5, 6],   # 4
                [3, 7],         # 5
                [0, 4],         # 6
                [1, 5],         # 7
                [2, 6],         # 8
                [4, 5, 6, 7],   # 9
                [0, 1, 2, 3],   # 10
                [4, 7],         # 11
                [0, 3],         # 12
                [1, 2],         # 13
                [5, 6],         # 14
                [6, 7],         # 15
                [4, 5],         # 16
                [0, 1],         # 17
                [2, 3]          # 18
            ],
            'axes': [
                [[0, 0], [0, 0], [0, 0], [0, 0]],
                [[1, 1], [1, 1], [1, 1], [1, 1]],
                [[0, 0], [0, 0], [0, 0], [0, 0]],
                [[1, 1], [1, 1], [1, 1], [1, 1]],
                [[0, 1], [0, 1]],
                [[0, 1], [0, 1]],
                [[0, 1], [0, 1]],
                [[0, 1], [0, 1]],
                [[2, 2], [2, 2], [2, 2], [2, 2]],
                [[2, 2], [2, 2], [2, 2], [2, 2]],
                [[1, 2], [1, 2]],
                [[1, 2], [1, 2]],
                [[1, 2], [1, 2]],
                [[1, 2], [1, 2]],
                [[0, 2], [0, 2]],
                [[0, 2], [0, 2]],
                [[0, 2], [0, 2]],
                [[0, 2], [0, 2]]
            ],
            'signs': [1, 1, 1, 1,
                      1, -1, 1, -1,
                      1, 1,
                      1, -1, 1, -1,
                      1, -1, 1, -1]
        }