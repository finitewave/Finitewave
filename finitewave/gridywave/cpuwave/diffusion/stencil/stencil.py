from abc import ABC, abstractmethod


class Stencil(ABC):
    """Base class for calculating stencil weights used in numerical
    simulations.

    This abstract base class defines the interface for calculating stencil
    weights for numerical simulations. It includes a caching mechanism to
    optimize performance by reducing the number of symbolic calculations. Also,
    it handles the boundary conditions for the numerical scheme.
    """
    @abstractmethod
    def assemble_matrices(self, simulation):
        """
        Assembles the stiffness and mass matrices based on the provided
        parameters.

        This method must be implemented by subclasses to compute the stiffness
        and mass matrices used for numerical simulations. The matrices are
        calculated based on the tissue mesh and spatial step. Additional
        parameters can be passed as arguments or keyword arguments.

        Parameters
        ----------
        simulation : simulation
            A simulation object containing the simulation parameters.

        Returns
        -------
        tuple of scipy.sparse.csr_matrix
            The stiffness and mass matrices.
        """
        pass
