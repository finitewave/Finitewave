from abc import ABC, abstractmethod
import copy


class DiffusionModel(ABC):
    """
    Base class for diffusion models.

    This class serves as the base for time integration. To speed up the
    simulation, it also adds the precomputed right-hand side (dt * I).

    Attributes
    ----------
    u : np.ndarray
        Array representing the action potential (mV) across the tissue.
        It can deviate from the model.u array.
    rhs : np.ndarray
        Array representing the right-hand side of the diffusion equation
        (dt * I).
    """
    def __init__(self):
        self.u = None
        self.rhs = None

    @abstractmethod
    def initialize(self, model):
        """
        Initializes the model with the given model.

        Parameters
        ----------
        model : CardiacModel
            The model object to initialize.
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
