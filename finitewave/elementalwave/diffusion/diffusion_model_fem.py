import numpy as np
import scipy.sparse.linalg as spla

from finitewave.core.diffusion.diffusion_model import DiffusionModel

from .solver.numba.crank_nicolson_cg_solver import (
    CrankNicolsonCGSolver as Solver
)
# from .solver.numba.implicit_euler_cg_solver import (
#     ImplicitEulerCGSolver as Solver
# )

from .assembler.triangle_assembler import TriangleAssembler
from .assembler.tetrahedral_assembler import TetrahedralAssembler


class DiffusionModelFEM(DiffusionModel):
    """
    A class to evaluate the diffusion part of the model using the
    Finite Element Method.

    Attributes
    ----------
    assembler : Assembler
        The assembler object used to assemble the stiffness and mass
        matrices.
    solver : Solver
        The solver object used to compute the diffusion. Default is
        CrankNicolsonCGSolver, which uses the Conjugate Gradient method
        from the scipy library.
    model : CardiacModel
        The instance of the CardiacModel class.
    u : array
        The potential values (contiguous array). It deviates from the
        model.u array to make it contiguous.
    rhs : array
        The right-hand side (contiguous array).
    matrices : array
        The matrices required by specific solver to evaluate the time step
        of the diffusion.
    """
    def __init__(self):
        super().__init__()
        self.assembler = None
        self.solver = Solver()
        self.simulation = None
        self.u = None
        self.rhs = None

    def initialize(self, simulation):
        """
        Initializes the diffusion model.

        Parameters
        ----------
        simulation : Simulation
            The simulation object containing the simulation parameters.
        """
        self.simulation = simulation

        if self.assembler is None:
            self.assembler = self.default_assembler(
                self.simulation.cardiac_tissue
            )

        self.compute_matrices()
        self.solver.initialize(self.simulation.cardiac_model.u)

    def compute_matrices(self):
        """
        Computes the stiffness and mass matrices required by the solver
        to evaluate the diffusion.
        """
        tissue = self.simulation.cardiac_tissue
        self.u = self.simulation.cardiac_model.u
        self.rhs = self.simulation.cardiac_model.rhs
        self.myo_indexes = self.simulation.cardiac_tissue.myo_indexes

        stiff, mass = self.assembler.assemble_matrices(self.simulation, tissue)
        self.matrices = self.solver.assemble_system(stiff, mass,
                                                    self.simulation.dt)

    def run(self):
        """
        Evaluates the diffusion part of the model.
        """
        self.u = self.simulation.cardiac_model.u
        self.u = self.solver.solve(self.u, self.rhs, self.myo_indexes,
                                   self.matrices)
        self.simulation.cardiac_model.u = self.u

    def default_assembler(self, cardiac_tissue):
        if cardiac_tissue.elems.shape[1] == 3:
            return TriangleAssembler()

        if cardiac_tissue.elems.shape[1] == 4:
            return TetrahedralAssembler()

        raise ValueError
