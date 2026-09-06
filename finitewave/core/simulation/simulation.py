import copy
from typing import Literal
import numpy as np


class Simulation:
    """Base class for cardiac simulations.

    This class stores simulation components and defines their initialization
    order.

    Attributes
    ----------
    cardiac_tissue : CardiacTissue
        The tissue object that represents the cardiac tissue in the simulation.
    stim_sequence : StimSequence
        The sequence of stimuli applied to the cardiac tissue.
    tracker_sequence : TrackerSequence
        The sequence of trackers used to monitor the simulation.
    command_sequence : CommandSequence
        The sequence of commands to execute during the simulation.
    state_loader : StateLoader
        The object responsible for loading the state of the simulation.
    state_saver : StateSaver
        The object responsible for saving the state of the simulation.
    time_integration : TimeIntegration
        The method used for time integration of the reaction-diffusion system.
    spatial_discretization : SpatialDiscretization
        The method used to assemble spatial operators.
    cardiac_model : CardiacModel
        The cardiac model that defines the ionic currents and state variables.
    dt : float
        Time step for the simulation.
    t_max : float
        Maximum time for the simulation (model units).
    t : float
        Current time in the simulation (model units).
    iteration : int
        Current step or iteration in the simulation.
    """
    def __init__(
            self,
            dt: float | None = None,
            t_max: float | None = None,
            backend: Literal["numba", "mlx", "jax"] = "numba",
            ):
        """Initialize simulation settings and empty component slots.

        Parameters
        ----------
        dt : float, optional
            Time step for the simulation. If None, it must be set before running.
        t_max : float, optional
            Maximum simulation time. If None, it must be set before running.
        backend : Literal["numba", "mlx", "jax"], optional
            Computational backend. Default is ``"numba"``.
        """

        self.meta = {}
        self.cardiac_tissue = None
        self.stim_sequence = None
        self.tracker_sequence = None
        self.command_sequence = None
        self.state_loader = None
        self.state_saver = None
        self.cardiac_model = None
        self.spatial_discretization = None
        self.time_integration = None

        self.dt = dt
        self.t_max = t_max
        self.t = 0
        self.iteration = 0
        self.backend = self.select_backend(backend)

    def select_backend(self, backend_name):
        """Select the computational backend for the simulation.

        Parameters
        ----------
        backend_name : str
            The name of the backend to use for computations.
        """
        raise NotImplementedError("Backend selection must be implemented in subclasses.")

    def initialize(self):
        """Initialize the model and attached simulation components.

        Sets up arrays, computes weights,
        and initializes stimuli, trackers, and commands.

        Note
        ----
        The order of initialization is important. The cardiac_model must be
        initialized before the spatial discretization and time integration,
        as they depend on the state variables of the cardiac model. Stimuli and
        trackers are initialized afterward because they may depend on the
        initialized model state.
        """
        self.iteration = 0
        self.t = 0
        self.cardiac_model.initialize(self)
        self.spatial_discretization.initialize(self)
        self.time_integration.initialize(self)

        if self.stim_sequence:
            self.stim_sequence.initialize(self)

        if self.tracker_sequence:
            self.tracker_sequence.initialize(self)

        if self.command_sequence:
            self.command_sequence.initialize(self)

        if self.state_loader:
            self.state_loader.initialize(self)

        if self.state_saver:
            self.state_saver.initialize(self)

    def run(self):
        """Run the simulation loop."""
        raise NotImplementedError

    def _remaining_steps(self):
        """Return the number of complete fixed-size steps before ``t_max``.

        Ratios sufficiently close to an integer are rounded to that integer to
        avoid losing a step to floating-point representation.
        """
        remaining = self.t_max - self.t
        if remaining <= 0:
            return 0

        ratio = remaining / self.dt
        nearest = round(ratio)
        if np.isclose(ratio, nearest, rtol=1e-12, atol=1e-12):
            return max(0, int(nearest))

        return max(0, int(np.floor(ratio)))

    def check_termination(self):
        """Check whether another complete time step can be performed.

        The simulation terminates at ``t_max`` or when the remaining duration
        is shorter than ``dt``. A ``CommandSequence`` may change ``t_max``
        during execution to control the simulation duration.

        Returns
        -------
        bool
            True if no complete step remains; otherwise, False.
        """
        return self._remaining_steps() == 0

    def clone(self):
        """Create a deep copy of this simulation.

        Returns
        -------
        Simulation
            A deep copy of this simulation instance.
        """
        return copy.deepcopy(self)
