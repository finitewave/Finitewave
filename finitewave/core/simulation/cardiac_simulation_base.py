import copy
import warnings
from tqdm import tqdm
import numpy as np
import numba


class CardiacSimulationBase:
    """
    Base class for electrophysiological models.

    This class serves as the base for implementing various cardiac models.
    It provides methods for initializing the model, running simulations,
    and managing the state of the simulation.

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
    dt : float
        Time step for the simulation.
    dr : float
        Spatial step for the simulation.
    t_max : float
        Maximum time for the simulation (model units).
    t : float
        Current time in the simulation (model units).
    step : int
        Current step or iteration in the simulation.
    prog_bar : bool
        Whether to display a progress bar during simulation.
    npfloat : type
        The floating-point type used for numerical computations.
    """
    def __init__(self):
        self.meta = {}
        self.cardiac_tissue = None
        self.stim_sequence = None
        self.tracker_sequence = None
        self.command_sequence = None
        self.state_loader = None
        self.state_saver = None

        self.solver = None
        self.diffusion_model = None
        self.cardiac_model = None

        self.dt = 0.
        self.dr = 0.
        self.t_max = 0.
        self.t = 0
        self.step = 0

        self.prog_bar = True
        self.npfloat = np.float64

    def initialize(self):
        """
        Initializes the model for simulation. Sets up arrays, computes weights,
        and initializes stimuli, trackers, and commands.
        """
        self.step = 0
        self.t = 0

        self.cardiac_model.initialize(self)
        self.diffusion_model.initialize(self)
        self.solver.initialize(self)

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
        """
        Runs the simulation loop. Handles stimuli, diffusion, ionic kernel
        updates, and tracking.
        """
        raise NotImplementedError

    def check_termination(self):
        """
        Checks whether the simulation should terminate based on the current
        time. The ``CommandSequence`` may change the ``t_max`` value during
        execution to control the simulation duration.

        Returns
        -------
        bool
            True if the simulation should terminate, False otherwise.
        """
        max_iters = int(np.ceil(self.t_max / self.dt))
        return (self.t > self.t_max) or (self.step > max_iters)

    def clone(self):
        """
        Creates a deep copy of the current model instance.

        Returns
        -------
        CardiacModel
            A deep copy of the current CardiacModel instance.
        """
        return copy.deepcopy(self)
