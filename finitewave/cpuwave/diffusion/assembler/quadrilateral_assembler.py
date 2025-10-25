import numpy as np
from .triangle_assembler import TriangleAssembler


class QuadrilateralAssembler(TriangleAssembler):
    def __init__(self):
        super().__init__()
        self.mass_coef = 36.  # consistent with 4-node quads

        self.elem_mass = 1 / self.mass_coef * np.array([[4, 2, 1, 2],
                                                        [2, 4, 2, 1],
                                                        [1, 2, 4, 2],
                                                        [2, 1, 2, 4]])

        self.dN_dxi = np.array([-0.25,  0.25,  0.25, -0.25])
        self.dN_deta = np.array([-0.25, -0.25,  0.25,  0.25])
        self.quad_weights = np.array([4.0])
        self.n_points = 4
