"""Abstract cardiac-model interface and plugin discovery."""

from abc import ABC, abstractmethod
import copy
from importlib.metadata import entry_points


class CardiacModelBase(ABC):
    """Abstract base class for cardiac reaction models.

    Finitewave models discover an operations plugin through the
    ``finitewave.models`` entry-point group and expose its state variables and
    parameters as model attributes.

    Attributes
    ----------
    u : ndarray
        Array representing the action potential (mV) across the tissue.
    rhs : ndarray
        Reaction term used by time integration.
    D_model : float
        Model-specific diffusion coefficient.
    state_vars : list
        List of model-specific state variables.
    state_pars : list
        List of model-specific parameters.
    simulation : Simulation
        Reference to the current simulation instance.
    observers : list
        Observer definitions evaluated by the generated reaction kernel.
    """
    model_name = None
    finitewave_model = True

    def __init__(self):
        """Initialize model metadata and load plugin operations when enabled."""
        self.u = None
        self.rhs = None
        self.D_model = None

        self.state_vars = []
        self.state_pars = []
        self.simulation = None

        self.observers = []

        if self.finitewave_model:
            self.ops = self.load_ops(self.model_name)
            self.initialize_variables_and_parameters()

    @abstractmethod
    def initialize(self, simulation):
        """Initialize the model for a simulation."""
        pass

    @abstractmethod
    def run(self):
        """Evaluate the model for one simulation time step."""
        pass

    def set_parameters(self, params):
        """Set model parameters used during the next initialization.

        Parameters
        ----------
        params : dict
            Mapping from parameter names to scalar or array values.

        Raises
        ------
        ValueError
            If a parameter name is unknown.
        """
        for name, value in params.items():
            if not hasattr(self, name):
                raise ValueError(f"Parameter '{name}' not found in the model.")
            setattr(self, name, value)

    def set_state_variables(self, init_vars):
        """Set state-variable values used during the next initialization.

        Parameters
        ----------
        init_vars : dict
            Mapping from state-variable names to new initial values.

        Raises
        ------
        ValueError
            If a state-variable name is unknown.
        """

        for name, value in init_vars.items():
            if not hasattr(self, f"init_{name}"):
                raise ValueError(f"Variable '{name}' not found in the model.")
            
            setattr(self, f"init_{name}", value)

    def _discover(self) -> dict:
        """Return installed Finitewave model entry points keyed by name."""
        eps = entry_points()
        group = "finitewave.models"
        if hasattr(eps, "select"):
            selected = eps.select(group=group)
        else:
            selected = eps.get(group, [])
        return {ep.name: ep for ep in selected}
    
    @abstractmethod
    def initialize_variables_and_parameters(self):
        """Expose variables and parameters provided by the operations plugin."""
        pass

    def load_ops(self, model_name: str):
        """Load and validate a Finitewave model operations plugin.
        
        Parameters
        ----------
        model_name : str
            The name of the model to load, which should correspond to an entry
            point in the ``finitewave.models`` group.

        Returns
        -------
        module
            Operations module providing ``get_variables``, ``get_parameters``,
            ``get_diffusion_coefficient``, and ``ionic_step``.

        Raises
        ------
        KeyError
            If no installed entry point matches ``model_name``.
        ValueError
            If the plugin does not provide a required operation.
        """
        REQS = ("get_variables", "get_parameters", "ionic_step")

        eps = self._discover()
        if model_name not in eps:
            raise KeyError(
                f"Model '{model_name}' not found via entry point group "
                "'finitewave.models'."
            )
        
        mod = eps[model_name].load()   # ops package
        ops = getattr(mod, "ops", mod)

        for name in REQS:
            if not hasattr(ops, name):
                raise ValueError(f"Model '{model_name}' missing '{name}' in ops.")
        return ops

    def clone(self):
        """Create a deep copy of this model.

        Returns
        -------
        CardiacModelBase
            Deep copy of this model instance.
        """
        return copy.deepcopy(self)
