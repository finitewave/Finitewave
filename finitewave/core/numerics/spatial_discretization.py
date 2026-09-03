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
    def compute_weights(self, tissue):
        """
        Computes the weights for the diffusion operator and mass matrix.

        Parameters
        ----------
        tissue : CardiacTissueBase
            The tissue object containing the mesh and diffusion tensor.

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
        stiffness, mass = self.compute_weights(self.simulation.cardiac_tissue)
        self.weights = (stiffness * D_model, mass)
