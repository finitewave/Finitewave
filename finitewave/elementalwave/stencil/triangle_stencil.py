import numpy as np
from scipy import sparse
from finitewave.elementalwave.solver.scipy.scipy_cg_solver import ScipyCGSolver
from finitewave.core.stencil import Stencil


class TriangleStencil(Stencil):
    def __init__(self):
        super().__init__()
        self.solver = ScipyCGSolver()
        self.D_ac = 1/9
        self.D_al = 1

    def compute_weights(self, model, tissue):
        coords = tissue.myo_coords
        elems = self.reindex_elems(tissue.coords,
                                   tissue.myo_elems,
                                   tissue.myo_indexes)

        diffusion = self.compute_diffusion(model, tissue)
        diffusion = diffusion[tissue.myo_elems_indexes]

        areas, grads = self.areas_and_grads(coords, elems)
        stiffness_matrix = self.assemble_stiffness_matrix(
            coords, elems, areas, grads, diffusion
        )
        mass_matrix = self.assemble_mass_matrix(coords, elems, areas)
        a_matrix = self.solver.axpy(mass_matrix, stiffness_matrix, model.dt)
        return a_matrix, mass_matrix

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
        diffusion *= model.D_model
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

    def assemble_stiffness_matrix(self, coords, elems, areas, grads, diffusion):
        rows, cols, data = [], [], []

        for e in range(elems.shape[0]):
            for i in range(3):
                for j in range(3):
                    rows.append(elems[e, i])
                    cols.append(elems[e, j])
                    val = areas[e] * (grads[e, i, :] @
                                      diffusion[e] @
                                      grads[e, j, :])
                    data.append(val)

        res = sparse.coo_matrix((data, (rows, cols)),
                                shape=(coords.shape[0], coords.shape[0]))
        return res.tocsr()

    def assemble_mass_matrix(self, coords, elems, areas):
        rows, cols, data = [], [], []
        for e in range(elems.shape[0]):
            Me = (areas[e] / 12.0) * (np.ones((3, 3)) + np.eye(3))
            for i in range(3):
                for j in range(3):
                    rows.append(elems[e, i])
                    cols.append(elems[e, j])
                    data.append(Me[i, j])

        res = sparse.coo_matrix((data, (rows, cols)),
                                shape=(coords.shape[0], coords.shape[0]))
        return res.tocsr()
