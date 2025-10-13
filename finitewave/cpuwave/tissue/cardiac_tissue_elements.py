import numpy as np
from finitewave.core.tissue.cardiac_tissue import CardiacTissue


class CardiacTissueElements(CardiacTissue):
    def __init__(self, coords, elems, elem_type="Tri"):
        super().__init__()
        self.coords = coords
        self.elems = elems
        self.mesh = np.ones(self.coords.shape[0])
        self.mesh_elems = np.ones(self.elems.shape[0])
        self.conductivity = 1.
        self.fibers = None
        self.meta["shape"] = elem_type
        self.meta["type"] = "Elements"
        self.meta["dim"] = coords.shape[1]

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

    @property
    def myo_indexes(self):
        return np.unique(self.myo_elems)

    @property
    def myo_elems_indexes(self):
        # TODO: Remove element islands
        myo_elems_mask = self.mesh_elems == 1
        myo_vertex_mask = self.mesh == 1
        myo_elems_mask &= np.all(myo_vertex_mask[self.elems], axis=1)
        return np.flatnonzero(myo_elems_mask)

    @property
    def myo_elems(self):
        return self.elems[self.myo_elems_indexes]

    @property
    def myo_coords(self):
        return self.coords[self.myo_indexes]
