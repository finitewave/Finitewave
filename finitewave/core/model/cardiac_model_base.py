from abc import ABC, abstractmethod
import copy


class CardiacModelBase(ABC):
    """
    Base class for electrophysiological models.

    This class serves as the base for implementing various cardiac models.
    It provides methods for initializing the model, running simulations,
    and managing the state of the simulation.

    Attributes
    ----------
    u : ndarray
        Array representing the action potential (mV) across the tissue.
    rhs : ndarray
        Array representing the sum of the ionic currents.
    D_model : float
        Model-specific diffusion coefficient.
    state_vars : list
        List of state variables used during simulation.
    state_pars : list
        List of state parameters used during simulation.
    """
    def __init__(self):
        self.u = None
        self.rhs = None
        self.D_model = None

        self.state_vars = []
        self.state_pars = []
        self.simulation = None

    @abstractmethod
    def initialize(self, simulation):
        pass

    @abstractmethod
    def run(self):
        pass

    def clone(self):
        """
        Creates a deep copy of the current model instance.

        Returns
        -------
        CardiacModelBase
            A deep copy of the current CardiacModel instance.
        """
        return copy.deepcopy(self)
