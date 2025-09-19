import numpy as np
from numba import njit, prange

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.femwave.stencil.triangle_stencil import (
    TriangleStencil
)

from finitewave.femwave.stencil.tetrahedral_stencil import (
    TetrahedralStencil
)


class CardiacModelElems(CardiacModel):
    def __init__(self):
        super().__init__()
        self.cardiac_tissue = None
        self.stencil = None
        self.stim_sequence = None
        self._rhs = np.ndarray
        self._u = np.ndarray
        self._u_new = np.ndarray

        self.npfloat = 'float64'
        self.init_u = 0

    def initialize(self):
        super().initialize()
        self.u[:] = self.init_u * np.ones_like(self.u)
        self._u = self.u[self.cardiac_tissue.myo_indexes].copy()
        self._rhs = np.zeros_like(self._u)
        self._u_new = self._u.copy()
        self.u_new = self.u

    def run(self, initialize=True, num_of_threads=None):
        super().run(initialize, num_of_threads)
        self.u[self.cardiac_tissue.myo_indexes] = self._u

    def swap_arrays(self):
        self._u_new, self._u = self._u, self._u_new

    def run_diffusion_kernel(self):
        """
        Executes the diffusion kernel computation using the current parameters
        and tissue weights.
        """
        self._u_new = self.diffusion_kernel(self._u_new, self._u, self._rhs,
                                            self.weights)

    def select_stencil(self, cardiac_tissue):
        if cardiac_tissue.elems.shape[1] == 3:
            return TriangleStencil()

        if cardiac_tissue.elems.shape[1] == 4:
            return TetrahedralStencil()
        raise ValueError
