from abc import abstractmethod
import numpy as np
from scipy import sparse
from numba import njit, prange
from finitewave.core.diffusion.diffusion_model_base import DiffusionModelBase


class ElementAssembler(DiffusionModelBase):
    """
    Class for assembling element-based diffusion models.

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
        self.reference_element = None
        self.simulation = None

    def initialize(self, simulation):
        """
        Computes the weights (stiffness and mass matrices) for the
        element-based model.

        Parameters
        ----------
        simulation : Simulation
            The simulation instance associated with this assembler.
        """
        self.simulation = simulation
        self.compute_weights()

    @abstractmethod
    def compute_metrics(self, coords, elems):
        """
        Computes the volumes/areas and gradients for each element.

        Parameters
        ----------
        coords : np.ndarray
            The coordinates of the mesh nodes.
        elems : np.ndarray
            The connectivity of the mesh elements.

        Returns
        -------
        np.ndarray
            The volumes/areas of the elements.
        np.ndarray
            The gradients of the basis functions for each element.
        """
        raise NotImplementedError

    def compute_weights(self):
        """
        Computes the stiffness and mass matrices for the element-based model.

        Returns
        -------
        scipy.sparse.csr_matrix
            The stiffness matrix for the element-based model.
        scipy.sparse.csr_matrix
            The mass matrix for the element-based model.
        """
        tissue = self.simulation.cardiac_tissue

        coords = tissue.myo_coords
        elems = self.reindex_elems(tissue.coords,
                                   tissue.myo_elems,
                                   tissue.myo_indexes)

        diffusion = self.compute_diffusion(self.simulation, tissue)
        diffusion = diffusion[tissue.myo_elems_indexes]

        volumes, grads = self.compute_metrics(coords, elems)
        self.weights = self.stiffness_and_mass_matrix(
            coords, elems, volumes, grads, diffusion,
            self.reference_element.elem_mass
        )
        return self.weights

    def compute_diffusion(self, simulation, tissue):
        """
        Computes the diffusion tensor for each element.

        Parameters
        ----------
        simulation : Simulation
            The simulation instance.
        tissue : CardiacTissue
            The cardiac tissue instance.

        Returns
        -------
        np.ndarray
            The diffusion tensor for each element.
        """
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
        """
        Reindexes the element connectivity array. Resulted indexes are
        continuous and start from 0.

        Parameters
        ----------
        coords : np.ndarray
            The coordinates of the mesh nodes.
        elems : np.ndarray
            The connectivity of the mesh elements.
        indexes : np.ndarray
            The indexes of the nodes in the original mesh.

        Returns
        -------
        np.ndarray
            The reindexed element connectivity array.
        """
        new_indexes = - np.ones(coords.shape[0], dtype=int)
        new_indexes[indexes] = np.arange(len(indexes))
        return new_indexes[elems]

    def stiffness_and_mass_matrix(self, coords, elems, volumes, grads,
                                  diffusion, elem_mass):
        """
        Computes the stiffness and mass matrices.
        Parameters
        ----------
        coords : np.ndarray
            The coordinates of the mesh nodes.
        elems : np.ndarray
            The connectivity of the mesh elements.
        volumes : np.ndarray
            The volumes/areas of the elements.
        grads : np.ndarray
            The gradients of the shape functions.
        diffusion : np.ndarray
            The diffusion tensors for each element.
        elem_mass : np.ndarray
            The mass matrix for each element.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            The rows, columns, stiffness matrix, and mass matrix.
        """
        shape = (coords.shape[0], coords.shape[0])
        rows, cols, stiff, mass = stiffness_and_mass_matrix(elems, volumes,
                                                            grads, diffusion,
                                                            elem_mass)
        stiffness_matrix = sparse.coo_matrix((stiff, (rows, cols)),
                                             shape=shape)
        mass_matrix = sparse.coo_matrix((mass, (rows, cols)), shape=shape)
        return stiffness_matrix.tocsr(), mass_matrix.tocsr()


@njit(parallel=True)
def stiffness_and_mass_matrix(elems, volumes, grads, diffusion, elem_mass):
    """
    Computes the stiffness and mass matrices.

    Parameters
    ----------
    elems : np.ndarray
        The connectivity of the mesh elements.
    volumes : np.ndarray
        The volumes/areas of the elements.
    grads : np.ndarray
        The gradients of the shape functions.
    diffusion : np.ndarray
        The diffusion tensors for each element.
    elem_mass : np.ndarray
        The mass matrix for each element.

    Returns
    -------
    np.ndarray
        The row indices of the stiffness and mass matrices.
    np.ndarray
        The column indices of the stiffness and mass matrices.
    np.ndarray
        The stiffness matrix data.
    np.ndarray
        The mass matrix data.
    """
    n_elems, n_points = elems.shape
    rows = np.zeros(n_elems * n_points ** 2, dtype=elems.dtype)
    cols = np.zeros_like(rows)
    stiff_data = np.zeros_like(rows, dtype=diffusion.dtype)
    mass_data = np.zeros_like(rows, dtype=diffusion.dtype)

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
                mass_data[ind] = volumes[e] * elem_mass[i, j]

    return rows, cols, stiff_data, mass_data
