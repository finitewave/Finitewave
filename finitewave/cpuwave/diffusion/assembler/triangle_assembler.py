import numpy as np

from .element_assembler import ElementAssembler


class TriangleAssembler(ElementAssembler):
    def __init__(self):
        super().__init__()
        self.mass_coef = 12.
        self.elem_mass = (1 / self.mass_coef) * np.array([[2, 1, 1],
                                                          [1, 2, 1],
                                                          [1, 1, 2]])
        # self.elem_mass = (1/3) * np.eye(3)
        self.dN_dxi = np.array([-1.0, 1.0, 0.0])
        self.dN_deta = np.array([-1.0, 0.0, 1.0])
        self.quad_weights = np.array([1/2])
        self.n_points = 3

    def volumes_and_grads(self, coords, elems):
        """
        Computes area and global gradients for linear triangles in 3D
        using the Jacobian from the reference triangle.

        Parameters:
        ----------
        coords: (N_nodes, 3)
            Coordinates of the mesh nodes.
        elems: (N_elems, 3)
            Element connectivity (node indices for each triangle).

        Returns:
        -------
        areas: (N_elems,)
            Area of each triangle element.
        grads: (N_elems, 3, 3)
            Gradient of shape functions in global coordinates for each element.
        """
        jacobian = self.build_jacobian(coords, elems)
        areas = self.compute_areas(jacobian)
        grads = self.compute_gradients(jacobian, self.n_points)

        return areas, grads

    def build_jacobian(self, coords, elems):
        """
        Build Jacobian matrices for quadrilateral elements.

        Parameters:
        ----------
        coords: (N_nodes, 3)
            Coordinates of the mesh nodes.
        elems: (N_elems, 4)
            Element connectivity (node indices for each quadrilateral).

        Returns:
        -------
        jacobian: (N_elems, 2, 3)
            Jacobian matrices for each quadrilateral element.
        """
        n_elems = elems.shape[0]
        jacobian = np.zeros((n_elems, 2, 3))

        for i in range(self.n_points):
            jacobian[:, 0, :] += self.dN_dxi[i] * coords[elems[:, i]]
            jacobian[:, 1, :] += self.dN_deta[i] * coords[elems[:, i]]

        return jacobian

    def compute_gradients(self, jacobian, n_points):
        """Compute global gradients for triangle elements.

        Parameters:
        ----------
        jacobian: (N, 2, 3)
            Jacobian matrices for N triangle elements.

        Returns:
        -------
            grads: (N, n_points, 3)
                Gradient of shape functions in global coordinates for each element.
        """
        jacobian_inv = self.invert_jacobian(jacobian)
        n_elems = jacobian_inv.shape[0]
        grads = np.zeros((n_elems, n_points, 3))

        for i in range(n_points):
            dN_ref = np.stack([np.full(n_elems, self.dN_dxi[i]),
                               np.full(n_elems, self.dN_deta[i])], axis=1)
            grads[:, i, :] = np.einsum('nij,nj->ni', jacobian_inv, dN_ref)

        return grads

    def compute_areas(self, jacobian):
        """Compute areas of triangles from their Jacobian matrices.

        Parameters:
        ----------
        jacobian: (N, 2, 3)
            Jacobian matrices for N triangle elements.

        Returns:
        -------
            areas: (N,)
                Area of each triangle element.
        """
        v1 = jacobian[:, 0, :]
        v2 = jacobian[:, 1, :]
        cross_v = np.cross(v1, v2)
        areas = self.quad_weights * np.linalg.norm(cross_v, axis=1)
        return areas

    def invert_jacobian(self, jacobian):
        """Compute the inverse of a 2x3 Jacobian using the pseudoinverse.

        Parameters:
        ----------
        jacobian: (N, 2, 3)
            Jacobian matrices for N elements.

        Returns:
        -------
            Jplus: (N, 3, 2)
                Right pseudoinverse of the Jacobian matrices.
        """
        JT = np.transpose(jacobian, (0, 2, 1))  # (N, 3, 2)
        G = np.matmul(jacobian, JT)             # (N, 2, 2)
        invG = np.linalg.inv(G)
        Jplus = np.matmul(JT, invG)  # (N, 3, 2) — right pseudoinverse
        return Jplus
