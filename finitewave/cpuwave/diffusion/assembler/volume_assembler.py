
import numpy as np
from .element_assembler import ElementAssembler


class VolumeAssembler(ElementAssembler):
    """
    Class for assembling volume element diffusion models.

    Attributes
    ----------
    reference_element : ReferenceElement
        The reference element used for numerical integration.
    simulation : Simulation
        The simulation instance associated with this assembler.
    weights : tuple
        The computed diffusion weights (stiffness and mass matrices).
    """
    def __init__(self):
        super().__init__()

    def compute_metrics(self, coords, elems):
        """
        Computes volume and global gradients for 3D elements
        using the Jacobian from the reference element.

        Parameters:
        ----------
        coords: (N_nodes, 3)
            Coordinates of the mesh nodes.
        elems: (N_elems, N_points)
            Element connectivity (node indices for each element).

        Returns:
        -------
        volumes: (N_elems,)
            Volume of each element.
        grads: (N_elems, 3, 3)
            Gradient of shape functions in global coordinates for each element.
        """
        jacobian = self.build_jacobian(coords, elems)
        volumes = self.compute_volumes(jacobian)
        grads = self.compute_gradients(jacobian)

        return volumes, grads

    def build_jacobian(self, coords, elems):
        """
        Build Jacobian matrices for quadrilateral elements.

        Parameters:
        ----------
        coords: (N_nodes, 3)
            Coordinates of the mesh nodes.
        elems: (N_elems, N_points)
            Element connectivity (node indices for each quadrilateral).

        Returns:
        -------
        jacobian: (N_elems, 3, 3)
            Jacobian matrices for each quadrilateral element.
        """
        n_elems = elems.shape[0]
        jacobian = np.zeros((n_elems, 3, 3))

        for i in range(self.reference_element.n_points):
            jacobian[:, 0, :] += (self.reference_element.dN_dxi[i] *
                                  coords[elems[:, i]])
            jacobian[:, 1, :] += (self.reference_element.dN_deta[i] *
                                  coords[elems[:, i]])
            jacobian[:, 2, :] += (self.reference_element.dN_dzeta[i] *
                                  coords[elems[:, i]])

        return jacobian

    def compute_gradients(self, jacobian):
        """Compute global gradients for triangle elements.

        Parameters:
        ----------
        jacobian: (N_elems, 3, 3)
            Jacobian matrices for N triangle elements.

        Returns:
        -------
            grads: (N_elems, N_points, 3)
                Gradient of shape functions in global coordinates for each
                element.
        """
        jacobian_inv = self.invert_jacobian(jacobian)
        n_elems = jacobian_inv.shape[0]
        n_points = self.reference_element.n_points
        grads = np.zeros((n_elems, n_points, 3))

        for i in range(n_points):
            dN_ref = np.stack(
                [np.full(n_elems, self.reference_element.dN_dxi[i]),
                 np.full(n_elems, self.reference_element.dN_deta[i]),
                 np.full(n_elems, self.reference_element.dN_dzeta[i])], axis=1
                )
            grads[:, i, :] = np.matmul(jacobian_inv, dN_ref[:, :, np.newaxis]
                                       ).squeeze()

        return grads

    def compute_volumes(self, jacobian):
        """Compute volumes of elements from their Jacobian matrices.

        Parameters:
        ----------
        jacobian: (N_elems, 3, 3)
            Jacobian matrices for N elements.

        Returns:
        -------
            volumes: (N_elems,)
                Volume of each element.
        """
        jacobian_det = np.abs(np.linalg.det(jacobian))
        return self.reference_element.quad_weights * jacobian_det

    def invert_jacobian(self, jacobian):
        """Invert Jacobian matrices.

        Parameters:
        ----------
        jacobian: (N_elems, 3, 3)
            Jacobian matrices for N elements.

        Returns:
        -------
            jacobian_inv: (N_elems, 3, 3)
                Inverted Jacobian matrices for N elements.
        """
        jacobian_inv = np.linalg.inv(jacobian)
        return jacobian_inv
