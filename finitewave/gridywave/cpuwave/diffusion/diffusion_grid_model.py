import numpy as np
from finitewave.core.diffusion.diffusion_model import DiffusionModel

from .stencil.isotropic_stencil import IsotropicStencil
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

    def compute_matrices(self):
        self.u_new = self.simulation.cardiac_model.u.copy()

        stiff, mass = self.stencil.assemble_matrices(self.simulation)
        self.matrices = self.solver.assemble_system(stiff, mass,
                                                    self.simulation.dt)

    def run(self, u, rhs, indexes):
        self.solver.solve(self.u_new, u, rhs, self.matrices, indexes)
        u, self.u_new = self.u_new, u
        return u

    def default_stencil(self):
        if self.simulation.cardiac_tissue.fibers is None:
            return IsotropicStencil()
