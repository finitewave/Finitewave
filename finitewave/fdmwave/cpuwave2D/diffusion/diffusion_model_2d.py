import numpy as np
from finitewave.core.diffusion.diffusion_model import DiffusionModel

from .stencil.isotropic_stencil_2d import IsotropicStencil2D
from .stencil.asymmetric_stencil_2d import AsymmetricStencil2D


class DiffusionModel2D(DiffusionModel):
    def __init__(self):
        super().__init__()
        self.weights = np.ndarray
        self.stencil = None
        self.simulation = None

    @property
    def u(self):
        return self.simulation.cardiac_model.u

    @u.setter
    def u(self, u):
        self.simulation.cardiac_model.u = u

    @property
    def rhs(self):
        return self.simulation.cardiac_model.rhs

    @rhs.setter
    def rhs(self, rhs):
        self.simulation.cardiac_model.rhs = rhs

    def initialize(self, simulation):
        super().initialize(simulation)
        self.simulation = simulation
        self.u_new = self.simulation.cardiac_model.u.copy()
        self.simulation.cardiac_tissue.compute_myo_indexes()

        if self.stencil is None:
            self.stencil = self.default_stencil()

        self.weights = self.stencil.compute_weights(simulation)

    def run(self):
        self.stencil.diffusion_kernel(self.u_new, self.u, self.rhs,
                                      self.weights,
                                      self.simulation.cardiac_tissue.myo_indexes)
        self.u_new, self.u = self.u, self.u_new

    def default_stencil(self):

        if self.simulation.cardiac_tissue.fibers is None:
            return IsotropicStencil2D()

        return AsymmetricStencil2D()
