import numpy as np
from numba import njit, prange

from finitewave.gridywave.cpuwave2D.model.aliev_panfilov_2d import (
    AlievPanfilov2D,
    ionic_kernel,
)
from finitewave.elementalwave.diffusion.diffusion_model_fem import (
    DiffusionModelFEM
)


class AlievPanfilovElems(AlievPanfilov2D):
    def __init__(self):
        super().__init__()
