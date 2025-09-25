import numpy as np
from scipy import sparse
from finitewave.femwave.solver.scipy.implicit_euler_cg_solver import (
    ImplicitEulerCGSolver
)
from finitewave.core.stencil import Stencil
from numba import njit, prange


class TriangleStencil(Stencil):
    def __init__(self):
        super().__init__()
        self.solver = ImplicitEulerCGSolver()
        self.D_ac = 1/9
        self.D_al = 1
        self.mass_coef = 12.

    def compute_weights(self, model, tissue):
        coords = tissue.myo_coords
        elems = self.reindex_elems(tissue.coords,
                                   tissue.myo_elems,
                                   tissue.myo_indexes)

        diffusion = self.compute_diffusion(model, tissue)
        diffusion = diffusion[tissue.myo_elems_indexes]

        areas, grads = self.areas_and_grads(coords, elems)
        stiffness_matrix, mass_matrix = self.stiffness_and_mass_matrix(
            coords, elems, areas, grads, diffusion, self.mass_coef
        )
        return self.solver.assemble_matrices(stiffness_matrix, mass_matrix,
                                             model.dt)

    def compute_diffusion(self, model, tissue):
        diffusion = np.eye(3, dtype=model.npfloat)

        if tissue.fibers is not None:
            diffusion = (self.D_ac * np.eye(3)[np.newaxis, :, :] +
                         ((self.D_al - self.D_ac) *
                          tissue.fibers[:, :, np.newaxis] @
                          tissue.fibers[:, np.newaxis, :]))

        conductivity = (model.D_model * tissue.conductivity *
                        np.ones(len(tissue.elems), dtype=model.npfloat))
        diffusion = diffusion * conductivity[:, np.newaxis, np.newaxis]
        return diffusion

    def select_diffusion_kernel(self):
        return self.solver.diffusion_kernel

    def reindex_elems(self, coords, elems, indexes):
        new_indexes = - np.ones(coords.shape[0], dtype=int)
        new_indexes[indexes] = np.arange(len(indexes))
        return new_indexes[elems]

    def areas_and_grads(self, coords, elems):
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

    def stiffness_and_mass_matrix(self, coords, elems, areas, grads, diffusion, mass_coef):
        # TODO: Lumped M
        rows, cols, stiff_data, mass_data = stiffness_and_mass_matrix(elems, areas, grads, diffusion, mass_coef)
        stiffness_matrix = sparse.coo_matrix((stiff_data, (rows, cols)),
                                             shape=(coords.shape[0], coords.shape[0]))
        mass_matrix = sparse.coo_matrix((mass_data, (rows, cols)),
                                        shape=(coords.shape[0], coords.shape[0]))
        return stiffness_matrix.tocsr(), mass_matrix.tocsr()


@njit(parallel=True)
def stiffness_and_mass_matrix(elems, areas, grads, diffusion, mass_coef):
    n_elems, n_points = elems.shape
    rows = np.zeros(n_elems * n_points ** 2, dtype=elems.dtype)
    cols = np.zeros_like(rows)
    stiff_data = np.zeros_like(rows, dtype=diffusion.dtype)
    mass_data = np.zeros_like(rows, dtype=diffusion.dtype)

    Me = (1 / mass_coef) * (np.ones((n_points, n_points)) +
                            np.eye(n_points))

    for e in prange(n_elems):
        for i in range(n_points):
            for j in range(n_points):
                ind = n_points ** 2 * (e - 1) + n_points * (i - 1) + j
                rows[ind] = elems[e, i]
                cols[ind] = elems[e, j]
                val = areas[e] * (grads[e, i, :] @
                                  diffusion[e] @
                                  grads[e, j, :])
                stiff_data[ind] = val
                mass_data[ind] = areas[e] * Me[i, j]

    return rows, cols, stiff_data, mass_data
