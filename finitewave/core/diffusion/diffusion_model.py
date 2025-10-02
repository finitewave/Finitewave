from abc import ABC, abstractmethod
import copy


class DiffusionModel(ABC):
    """
    Base class for diffusion models.

    This class serves as the base for time integration. To speed up the
    simulation, it also adds the precomputed right-hand side (dt * I).

    Attributes
    ----------
    simulation : simulation
        The simulation object to which the model is attached.
    """
    def __init__(self):
        self.simulation = None

    @abstractmethod
    def initialize(self, simulation):
        """
        Initializes the model with the given simulation.

        Parameters
        ----------
        simulation : simulation
            The simulation object to which the model is attached.
        """
        pass

    @abstractmethod
    def run(self):
        """
        Evaluates the diffusion part of the model.
        """
        pass

    def update_output(self):
        """
        Updates the output of the model.
        """
        pass

    def clone(self):
        """
        Creates a deep copy of the current model instance.

        Returns
        -------
        CardiacModel
            A deep copy of the current CardiacModel instance.
        """
        return copy.deepcopy(self)
