
import numpy as np
from finitewave.femwave.stencil.triangle_stencil import TriangleStencil


class TetrahedralStencil(TriangleStencil):
    def __init__(self):
        super().__init__()
        self.mass_coef = 20

    def areas_and_grads(self, coords, elems):
        p0 = coords[elems[:, 0]]
        p1 = coords[elems[:, 1]]
        p2 = coords[elems[:, 2]]
        p3 = coords[elems[:, 3]]

        v0 = p1-p0
        v1 = p2-p0
        v2 = p3-p0

        cross = np.cross(v1, v2)
        dot = np.sum(v0 * cross, axis=1)
        volumes = np.abs(dot) / 6

        B = np.stack([v0, v1, v2], axis=2)
        B_inv = np.linalg.inv(B)
        grad_1_to_3 = B_inv
        grad_0 = -np.sum(B_inv, axis=1)
        grads = np.concatenate([grad_0[:, None, :], grad_1_to_3], axis=1)

        return volumes, grads
