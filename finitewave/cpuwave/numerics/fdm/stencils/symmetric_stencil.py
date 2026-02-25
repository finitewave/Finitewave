import numpy as np
from .stecil import Stencil


class SymmetricStencil(Stencil):
    def __init__(self):
        pass

    def compute_flux_weights(self, mesh, diffusion, dr, ijk):
        """
        2 - 3
        |   |
        0 - 1
        """
        # qx = Dxx * du/dx + Dxy * du/dy
        # qy = Dyx * du/dx + Dyy * du/dy
        ijk_0 = ijk
        ijk_1 = self.build_neighbor(ijk, shift=1, axis=0)
        ijk_2 = self.build_neighbor(ijk, shift=1, axis=1)
        ijk_3 = self.build_neighbor(ijk_1, shift=1, axis=1)

        cell_is_valid = (self.is_valid_index(ijk_0, mesh) &
                         self.is_valid_index(ijk_1, mesh) &
                         self.is_valid_index(ijk_2, mesh) &
                         self.is_valid_index(ijk_3, mesh))
        
        dxx = diffusion[*ijk, 0, 0]
        dxy = diffusion[*ijk, 0, 1]
        dyx = diffusion[*ijk, 1, 0]
        dyy = diffusion[*ijk, 1, 1]

        wx = [- dxx / (2 * dr) - dxy / (2 * dr),
                dxx / (2 * dr) - dxy / (2 * dr),
              - dxx / (2 * dr) + dxy / (2 * dr),
                dxx / (2 * dr) + dxy / (2 * dr)]
        wy = [- dyx / (2 * dr) - dyy / (2 * dr),
                dyx / (2 * dr) - dyy / (2 * dr),
              - dyx / (2 * dr) + dyy / (2 * dr),
                dyx / (2 * dr) + dyy / (2 * dr)]


        wx = [- np.where(cell_is_valid, w, 0) for w in wx]
        wy = [- np.where(cell_is_valid, w, 0) for w in wy]
        ijk_list = [ijk_0, ijk_1, ijk_2, ijk_3]

        return ijk_list, wx, wy
    
    def compute_diffusion_weights(self, mesh, diffusion, dr, indexes):
        ijk = np.array(np.unravel_index(indexes, mesh.shape))
        ijk_list, wx_list, wy_list = self.compute_flux_weights(mesh, diffusion, dr, ijk)
        
        directions = [(1, 1), (-1, 1), (1, -1), (-1, -1)]

        rows, cols, weights = [], [], []
        for ijk, dirs in zip(ijk_list, directions):
            rx, cx, wx = self.nonzero_weights(mesh, ijk, ijk_list, wx_list, dirs[0])
            ry, cy, wy = self.nonzero_weights(mesh, ijk, ijk_list, wy_list, dirs[1])
            rows += rx + ry
            cols += cx + cy
            weights += wx + wy

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        weights = np.concatenate(weights) / (2 * dr)
        return rows, cols, weights


