import numpy as np

from .assembler import Assembler


class TriangleAssembler(Assembler):
    def __init__(self):
        super().__init__()
        self.mass_coef = 12.

    def volumes_and_grads(self, coords, elems):
        grads = np.zeros((elems.shape[0], 3, 3))

        # vertice 0
        p0 = coords[elems[:, 0]]
        p1 = coords[elems[:, 1]]
        p2 = coords[elems[:, 2]]

        normals = np.cross(p1 - p0, p2 - p0)
        areas = 0.5 * np.linalg.norm(normals, axis=1)

        phi_0 = np.cross(p1 - p2, normals) / (2.0 * areas[:, np.newaxis]) ** 2
        phi_1 = np.cross(p2 - p0, normals) / (2.0 * areas[:, np.newaxis]) ** 2
        phi_2 = np.cross(p0 - p1, normals) / (2.0 * areas[:, np.newaxis]) ** 2

        grads[:, 0, :] = phi_0
        grads[:, 1, :] = phi_1
        grads[:, 2, :] = phi_2

        return areas, grads
