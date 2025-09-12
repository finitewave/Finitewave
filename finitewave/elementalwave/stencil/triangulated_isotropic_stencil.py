import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from finitewave.core.stencil.stencil import Stencil


class TriangulatedIsotropicStencil(Stencil):
    def __init__(self):
        super().__init__()

    def compute_weights(self, model, cardiac_tissue):
        coords = cardiac_tissue.coords
        elems = cardiac_tissue.elements
        dt = model.dt
        diffusion = model.D_model * cardiac_tissue.conductivity

        areas, grads = self.areas_and_gradients(coords, elems)
        stiffness_matrix = self.stiffness_matrix(coords, elems, areas, grads,
                                                 diffusion)
        mass_matrix = self.mass_matrix(coords, elems, areas)
        a_matrix = self.build_a_matrix(stiffness_matrix, mass_matrix, dt)
        return a_matrix, mass_matrix

    def select_diffusion_kernel(self):
        return diffusion_kernel_scipy

    def areas_and_gradients(self, coords, elems):
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

    def stiffness_matrix(self, coords, elems, areas, grads, diffusion):
        rows = []
        cols = []
        data = []

        for e in range(elems.shape[0]):
            for i in range(3):
                for j in range(3):
                    rows.append(elems[e, i])
                    cols.append(elems[e, j])
                    val = diffusion * areas[e] * np.dot(grads[e, i, :],
                                                        grads[e, j, :])
                    data.append(val)

        res = sparse.coo_matrix((data, (rows, cols)),
                                shape=(coords.shape[0], coords.shape[0]))
        return res.tocsr()

    def mass_matrix(self, coords, elems, areas):
        # TODO: Lumped version
        rows = []
        cols = []
        data = []

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

    def build_a_matrix(self, stiffness_matrix, mass_matrix, dt):
        return mass_matrix + dt * stiffness_matrix


def diffusion_kernel_scipy(
        u_new,
        u,
        a_matrix,
        mass_matrix,
        rhs,
        dt,
        atol=1e-6):
    # TODO: Preconditioner
    b = mass_matrix.dot(u, dt * rhs)
    u_new[:], n_iter = linalg.cg(a_matrix, b, atol=atol)

    if n_iter > 0:
        print("Convergence to tolerance not achieved")

    return u_new
