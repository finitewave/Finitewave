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
    def compute_weights(self, model, cardiac_tissue):
        """
        Computes the stencil weights based on the provided parameters.

        This method must be implemented by subclasses to compute the stencil
        weights used for numerical simulations. The weights are calculated
        based on the tissue mesh and spatial step. Additional parameters can
        be passed as arguments or keyword arguments.

        Parameters
        ----------
        model : CardiacModel
            A model object containing the simulation parameters.
        cardiac_tissue : CardiacTissue
            A tissue object representing the cardiac tissue.

        Returns
        -------
        np.ndarray
            A numpy array containing the stencil weights.
        """
        pass

    @abstractmethod
    def select_diffusion_kernel():
        """
        Builds the diffusion kernel for the numerical scheme.

        This method must be implemented by subclasses to build the diffusion
        kernel used for the numerical scheme. The kernel is used to compute the
        diffusion of the potential in the tissue mesh.
        """
        pass
