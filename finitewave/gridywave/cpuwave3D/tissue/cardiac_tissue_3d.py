import numpy as np
from finitewave.gridywave.cpuwave2D.tissue.cardiac_tissue_2d import (
    CardiacTissue2D
)


class CardiacTissue3D(CardiacTissue2D):
    def __init__(self, shape):
        super().__init__(shape)

    def add_boundaries(self):
        self.mesh[0, :, :] = 0
        self.mesh[-1, :, :] = 0
        self.mesh[:, 0, :] = 0
        self.mesh[:, -1, :] = 0
        self.mesh[:, :, 0] = 0
        self.mesh[:, :, -1] = 0

    def compute_myo_indexes(self):
        self.myo_indexes = np.flatnonzero(self.mesh == 1)
        self.tissue_indexes = np.flatnonzero(self.mesh > 0)
        tissue_mesh = self.mesh[self.mesh > 0]
        self.myo_tissue_indexes = np.flatnonzero(tissue_mesh == 1)
