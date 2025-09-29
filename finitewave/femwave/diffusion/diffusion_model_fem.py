import numpy as np

from finitewave.core.diffusion.diffusion_model import DiffusionModel

from .solver.scipy.crank_nicolson_cg_solver import (
    CrankNicolsonCGSolver
)

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
        self.solver = CrankNicolsonCGSolver()
        self.model = None
        self.u = None
        self.rhs = None

    def initialize(self, model):
        """
        Initializes the diffusion model.
        """
        self.model = model
        self.model.cardiac_tissue.compute_myo_indexes()

        tissue = self.model.cardiac_tissue
        self.u = self.model.u[tissue.myo_indexes].copy()
        self.rhs = np.zeros_like(self.u)

        if self.assembler is None:
            self.assembler = self.default_assembler(tissue)

        stiff, mass = self.assembler.assemble_matrices(self.model, tissue)
        self.matrices = self.solver.assemble_system(stiff, mass, self.model.dt)

    def update_state(self):
        """
        Updates the state of the model.
        """
        self.u = self.model.u[self.model.cardiac_tissue.myo_indexes].copy()

    def update_output(self):
        """
        Updates the output of the model.
        """
        self.model.u[self.model.cardiac_tissue.myo_indexes] = self.u

    def run(self):
        """
        Evaluates the diffusion part of the model.
        """
        self.u = self.solver.solve(self.u, self.rhs, self.matrices)

    def default_assembler(self, cardiac_tissue):
        if cardiac_tissue.elems.shape[1] == 3:
            return TriangleAssembler()

        if cardiac_tissue.elems.shape[1] == 4:
            return TetrahedralAssembler()

        raise ValueError
