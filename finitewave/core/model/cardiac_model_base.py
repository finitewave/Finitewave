from abc import ABC, abstractmethod
import copy
from importlib.metadata import entry_points


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
    step : int
        Frequency of calling the ionic kernel (in terms of time steps).
    iter_counter : int
        Counter to track the number of simulation iterations.
    state_vars : list
        List of model-specific state variables.
    state_pars : list
        List of model-specific parameters.
    simulation : Simulation
        Reference to the current simulation instance.
    observers : list
        List of observer functions to be called inside the `ionic_kernel`.
    """
    model_name = None
    finitewave_model = True

    def __init__(self):
        self.u = None
        self.rhs = None
        self.D_model = None

        self.step = 1
        self.iter_counter = 0

        self.state_vars = []
        self.state_pars = []
        self.simulation = None

        self.observers = []

        if self.finitewave_model:
            self.ops = self.load_ops(self.model_name)
            self.initialize_variables_and_parameters()

    @abstractmethod
    def initialize(self, simulation):
        pass

    @abstractmethod
    def run(self):
        pass

    def set_parameters(self, params):
        """
        Updates the model's parameters with the provided values.

        Parameters
        ----------
        params : dict
            Dictionary of parameter names and their new values.
        """
        for name, value in params.items():
            if not hasattr(self, name):
                raise ValueError(f"Parameter '{name}' not found in the model.")
            setattr(self, name, value)

    def set_state_variables(self, init_vars):
        """
        Updates the model's initial values for the state variables.

        Parameters
        ----------
        init_vars : dict
            Dictionary of variable names and their new initial values.
        initial : bool, optional
            Whether the provided values are initial conditions (default is False).
            If True, the values will be set to `init_{var}` attributes.
            If False, they will be set to the current state variable arrays.
        """

        for name, value in init_vars.items():
            if not hasattr(self, f"init_{name}"):
                raise ValueError(f"Variable '{name}' not found in the model.")
            
            setattr(self, f"init_{name}", value)

    def _discover(self) -> dict:
        eps = entry_points()
        group = "finitewave.models"
        if hasattr(eps, "select"):
            selected = eps.select(group=group)
        else:
            selected = eps.get(group, [])
        return {ep.name: ep for ep in selected}
    
    @abstractmethod
    def initialize_variables_and_parameters(self):
        pass

    def load_ops(self, model_name: str):
        """
        Loads the finitewave model.
        
        Parameters
        ----------
        model_name : str
            The name of the model to load, which should correspond to an entry
            point in the 'finitewave.models' group.

        Returns
        -------
        module
            The operations module for the specified model, containing necessary
            functions for model execution.
        """
        REQS = ("get_variables", "get_parameters", "ionic_step")

        eps = self._discover()
        if model_name not in eps:
            raise KeyError(f"Model '{model_name}' not found via entry point group 'finitewave.models'.")
        
        mod = eps[model_name].load()   # ops package
        ops = getattr(mod, "ops", mod)

        for name in REQS:
            if not hasattr(ops, name):
                raise ValueError(f"Model '{model_name}' missing '{name}' in ops.")
        return ops

    def clone(self):
        """
        Creates a deep copy of the current model instance.

        Returns
        -------
        CardiacModelBase
            A deep copy of the current CardiacModel instance.
        """
        return copy.deepcopy(self)
