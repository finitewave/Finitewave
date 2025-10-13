import numpy as np
from scipy import sparse
from numba import njit, prange


class Assembler:
    def __init__(self):
        super().__init__()
        self.mass_coef = None

    def assemble_matrices(self, simulation):
        tissue = simulation.cardiac_tissue

        coords = tissue.myo_coords
        elems = self.reindex_elems(tissue.coords,
                                   tissue.myo_elems,
                                   tissue.myo_indexes)

        diffusion = self.compute_diffusion(simulation, tissue)
        diffusion = diffusion[tissue.myo_elems_indexes]

        volumes, grads = self.volumes_and_grads(coords, elems)
        stiffness_matrix, mass_matrix = self.stiffness_and_mass_matrix(
            coords, elems, volumes, grads, diffusion, self.mass_coef
        )
        return stiffness_matrix, mass_matrix

    def compute_diffusion(self, simulation, tissue):
        d_ac = tissue.D_ac
        d_al = tissue.D_al
        d_model = simulation.cardiac_model.D_model

        diffusion = np.eye(3, dtype=simulation.npfloat)

        if tissue.fibers is not None:
            diffusion = (d_ac * np.eye(3)[np.newaxis, :, :] +
                         ((d_al - d_ac) *
                          tissue.fibers[:, :, np.newaxis] @
                          tissue.fibers[:, np.newaxis, :]))

        conductivity = (d_model * tissue.conductivity *
                        np.ones(len(tissue.elems), dtype=simulation.npfloat))
        diffusion = diffusion * conductivity[:, np.newaxis, np.newaxis]
        return diffusion

    def reindex_elems(self, coords, elems, indexes):
        new_indexes = - np.ones(coords.shape[0], dtype=int)
        new_indexes[indexes] = np.arange(len(indexes))
        return new_indexes[elems]

    def stiffness_and_mass_matrix(self, coords, elems, volumes, grads,
                                  diffusion, mass_coef):
        # TODO: Lumped M
        shape = (coords.shape[0], coords.shape[0])
        rows, cols, stiff, mass = stiffness_and_mass_matrix(elems, volumes,
                                                            grads, diffusion,
                                                            mass_coef)
        stiffness_matrix = sparse.coo_matrix((stiff, (rows, cols)),
                                             shape=shape)
        mass_matrix = sparse.coo_matrix((mass, (rows, cols)), shape=shape)
        return stiffness_matrix.tocsr(), mass_matrix.tocsr()


@njit(parallel=True)
def stiffness_and_mass_matrix(elems, volumes, grads, diffusion, mass_coef):
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
                val = volumes[e] * (grads[e, i, :] @
                                    diffusion[e] @
                                    grads[e, j, :])
                stiff_data[ind] = val
                mass_data[ind] = volumes[e] * Me[i, j]

    return rows, cols, stiff_data, mass_data
