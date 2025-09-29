import numpy as np
from finitewave.core.diffusion.diffusion_model import DiffusionModel

from .stencil.isotropic_stencil_2d import IsotropicStencil2D
from .stencil.asymmetric_stencil_2d import AsymmetricStencil2D


class DiffusionModel2D(DiffusionModel):
    def __init__(self):
        super().__init__()
        self.weights = np.ndarray
        self.stencil = None

    def initialize(self, model):
        super().initialize(model)
        self.u_new = self.model.u.copy()
        self.model.cardiac_tissue.compute_myo_indexes()

        if self.stencil is None:
            self.stencil = self.default_stencil(model.cardiac_tissue)

        self.weights = self.stencil.compute_weights(self, model.cardiac_tissue)

    def run(self):
        self.stencil.diffusion_kernel(self.u_new, self.model.u, self.rhs,
                                      self.weights,
                                      self.model.cardiac_tissue.myo_indexes)
        self.u_new, self.model.u = self.model.u, self.u_new

    def default_stencil(self, cardiac_tissue):

        if cardiac_tissue.fibers is None:
            return IsotropicStencil2D()

        return AsymmetricStencil2D()
