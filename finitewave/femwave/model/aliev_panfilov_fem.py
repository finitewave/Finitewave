import numpy as np
from numba import njit, prange

from finitewave.fdmwave.cpuwave2D.model.aliev_panfilov_2d import (
    AlievPanfilov2D,
    ionic_kernel,
)
from finitewave.femwave.diffusion.diffusion_model_fem import (
    DiffusionModelFEM
)


class AlievPanfilovFEM(AlievPanfilov2D):
    def __init__(self):
        super().__init__()
        self.diffusion_model = DiffusionModelFEM()

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel for the Aliev-Panfilov model.
        """
        ionic_kernel(self.diffusion_model.u, self.diffusion_model.rhs,
                     self.cardiac_tissue.myo_indexes, self.dt, self.v, self.a,
                     self.k, self.eap, self.mu_1, self.mu_2,
                     continuous_indexing=True)
