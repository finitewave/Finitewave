from abc import ABC, abstractmethod


class SpatialDiscretization(ABC):
    """
    Base class for spatial discretization methods.
    """

    def initialize(self, simulation):
        """
        Initializes the spatial discretization method.
        """
        self.simulation = simulation
        self.update_weights()

    @abstractmethod
    def compute_weights(self, tissue, D_model=1.):
        """
        Computes the weights for the diffusion operator and mass matrix.

        Parameters
        ----------
        tissue : CardiacTissueBase
            The tissue object containing the mesh and diffusion tensor.
        D_model : float, optional
            The diffusion coefficient to scale the stiffness matrix, by default 1.

        Returns
        -------
        sparse.csr_matrix
            The stiffness matrix with shape (non_empty_nodes, non_empty_nodes).
        sparse.csr_matrix
            The mass matrix with shape (non_empty_nodes, non_empty_nodes).
        """
        pass

    def update_weights(self):
        D_model = self.simulation.cardiac_model.D_model
        tissue = self.simulation.cardiac_tissue
        stiffness, mass = self.compute_weights(tissue, D_model)
        self.weights = (stiffness, mass)
