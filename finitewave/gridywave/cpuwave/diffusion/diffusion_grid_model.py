import numpy as np
from finitewave.core.diffusion.diffusion_model import DiffusionModel

from .stencil.isotropic_stencil import IsotropicStencil
from .stencil.anisotropic_stencil import AnisotropicStencil
from .solver.explicit_euler import ExplicitEuler


class DiffusionGridModel(DiffusionModel):
    def __init__(self):
        super().__init__()
        self.solver = ExplicitEuler()
        self.stencil = None
        self.simulation = None

    def initialize(self, simulation):
        super().initialize(simulation)
        self.simulation = simulation
        self.init_variables()

        if self.stencil is None:
            self.stencil = self.default_stencil()

        self.compute_matrices()

    def init_variables(self):
        self.u_new = self.simulation.cardiac_model.u.copy()
        self.u = self.simulation.cardiac_model.u
        self.rhs = self.simulation.cardiac_model.rhs
        self.myo_indexes = self.simulation.cardiac_model.myo_indexes

    def compute_matrices(self):
        stiff, mass = self.stencil.assemble_matrices(self.simulation)
        self.matrices = self.solver.assemble_system(stiff, mass,
                                                    self.simulation.dt)

    def run(self):
        self.solver.solve(self.u_new, self.u, self.rhs, self.myo_indexes,
                          self.matrices)
        self.u, self.u_new = self.u_new, self.u

    def default_stencil(self):
        if self.simulation.cardiac_tissue.fibers is None:
            return IsotropicStencil()

        return AnisotropicStencil()
