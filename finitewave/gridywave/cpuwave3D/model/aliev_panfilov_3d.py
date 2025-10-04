import numpy as np
from finitewave.gridywave.cpuwave2D.model.aliev_panfilov_2d import (
    AlievPanfilov2D,
    ionic_kernel
)


class AlievPanfilov3D(AlievPanfilov2D):
    """
    This class implements the Aliev–Panfilov model of cardiac excitation in 3D.
    """
    def __init__(self, memory_save=False):
        super().__init__()
        self.memory_save = memory_save

    def initialize(self, simulation):
        self.simulation = simulation
        mesh = self.simulation.cardiac_tissue.mesh
        self.u = self.init_u * np.ones(mesh.shape, dtype=simulation.npfloat)
        self.rhs = np.zeros_like(self.u)
        self.init_state_vars()

    def init_state_vars(self):
        for var in self.state_vars:
            shape = len(self.simulation.cardiac_tissue.tissue_indexes)
            init_val = getattr(self, "init_" + var)
            val = init_val * np.ones(shape, dtype=self.simulation.npfloat)
            setattr(self, "_" + var, val)

    def __getattr__(self, name):
        if name in self.state_vars:
            val = getattr(self, "_" + name)
            return self.restore_shape(val)

    def __setattr__(self, name, value):
        if "state_vars" in self.__dict__ and name in self.state_vars:
            value = self.reduce_shape(value)
            setattr(self, "_" + name, value)
            return
        super().__setattr__(name, value)

    def reduce_shape(self, value):
        if self.memory_save:
            return value.flat[self.simulation.cardiac_tissue.tissue_indexes]
        return value

    def restore_shape(self, value):
        if not self.memory_save:
            return value
        out = np.zeros_like(self.u)
        out.flat[self.simulation.cardiac_tissue.tissue_indexes] = value
        return out

    def run(self):
        if self.memory_save:
            ionic_kernel(self.u, self.rhs,
                         self.simulation.cardiac_tissue.myo_tissue_indexes,
                         self.simulation.dt,
                         self._v, self.a, self.k, self.eap, self.mu_1,
                         self.mu_2, self.simulation.cardiac_tissue.myo_indexes)

            return

        ionic_kernel(self.u, self.rhs,
                     self.simulation.cardiac_tissue.myo_indexes,
                     self.simulation.dt,
                     self.v, self.a, self.k, self.eap, self.mu_1,
                     self.mu_2)
