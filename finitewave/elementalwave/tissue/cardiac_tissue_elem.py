import numpy as np
from finitewave.core.tissue.cardiac_tissue import CardiacTissue


class CardiacTissueElem(CardiacTissue):
    def __init__(self, coords, elems):
        super().__init__()
        self.coords = coords
        self.elems = elems
        self.mesh = np.ones(self.coords.shape[0])
        self.mesh_elems = np.ones(self.elems.shape[0])
        self.conductivity = 1.

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, coords):
        if coords.shape[1] == 2:
            coords = np.hstack([coords, np.zeros(coords.shape[0])])
        self._coords = coords

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        self._mesh = mesh

    def add_boundaries(self):
        pass

    def compute_myo_indexes(self):
        self.myo_indexes = np.unique(self.myo_elems)

    @property
    def myo_elems(self):
        myo_elems = self.elems[self.mesh_elems == 1]
        vertex_mask = self.mesh == 1
        elem_mask = np.all(vertex_mask[myo_elems], axis=1)
        return myo_elems[elem_mask]

    @property
    def myo_coords(self):
        self.compute_myo_indexes()
        return self.coords[self.myo_indexes]
