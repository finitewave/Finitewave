import numpy as np
from finitewave.core.diffusion.diffusion_model import DiffusionModel

from .stencil2D.isotropic_stencil_2d import IsotropicStencil2D
from .stencil2D.asymmetric_stencil_2d import AsymmetricStencil2D
from .stencil3D.isotropic_stencil_3d import IsotropicStencil3D
from .stencil3D.asymmetric_stencil_3d import AsymmetricStencil3D


class DiffusionGridModel(DiffusionModel):
    def __init__(self):
        super().__init__()
        self.weights = np.ndarray
        self.stencil = None
        self.simulation = None

    def initialize(self, simulation):
        super().initialize(simulation)
        self.simulation = simulation
        self.init_variables()

        if self.stencil is None:
            self.stencil = self.default_stencil()

        self.compute_weights()

    def init_variables(self):
        if self.simulation.cardiac_model.memory_save:
            self.u = np.zeros_like(self.simulation.cardiac_tissue.mesh,
                                   dtype=self.simulation.npfloat)
            self.u.flat[:] = self.simulation.cardiac_model.init_u
            self.u_new = self.u.copy()
            self.rhs = np.zeros_like(self.u)
            return

        self.u = self.simulation.cardiac_model.u
        self.u_new = self.u.copy()
        self.rhs = np.zeros_like(self.u)

    def compute_weights(self):
        self.weights = self.stencil.compute_weights(self.simulation)
        self.myo_indexes = self.simulation.cardiac_tissue.myo_indexes

    def run(self):
        self.stencil.diffusion_kernel(self.u_new, self.u, self.rhs,
                                      self.weights, self.myo_indexes)
        self.u_new, self.u = self.u, self.u_new

    def default_stencil(self):

        if self.simulation.cardiac_tissue.fibers is None:
            if self.simulation.cardiac_tissue.mesh.ndim == 2:
                return IsotropicStencil2D()
            return IsotropicStencil3D()

        if self.simulation.cardiac_tissue.mesh.ndim == 2:
            return AsymmetricStencil2D()
        return AsymmetricStencil3D()
