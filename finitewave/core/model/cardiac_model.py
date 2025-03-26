from abc import ABC, abstractmethod
import copy
from tqdm import tqdm
import numpy as np
import numba


class CardiacModel(ABC):
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
    stencil : Stencil
        The stencil used for numerical computations.
    u : ndarray
        Array representing the action potential (mV) across the tissue.
    u_new : ndarray
        Array for storing the updated action potential values.
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
    D_model : float
        Model-specific diffusion coefficient.
    prog_bar : bool
        Whether to display a progress bar during simulation.
    npfloat : type
        The floating-point type used for numerical computations.
    state_vars : list
        List of state variables to save and load during simulation.
    """
    def __init__(self):
        self.meta = {}
        self.cardiac_tissue = None
        self.stim_sequence = None
        self.tracker_sequence = None
        self.command_sequence = None
        self.state_loader = None
        self.state_saver = None
        self.stencil = None

        self.diffuse_kernel = None
        self.ionic_kernel = None

        self.u = np.ndarray
        self.u_new = np.ndarray
        self.dt = 0.
        self.dr = 0.
        self.t_max = 0.
        self.t = 0
        self.step = 0
        self.D_model = 1.

        self.prog_bar = True
        self.npfloat = np.float64
        self.state_vars = []

    @abstractmethod
    def run_ionic_kernel(self):
        """
        Abstract method for running the ionic kernel. Must be implemented by
        subclasses.
        """
        pass

    def initialize(self):
        """
        Initializes the model for simulation. Sets up arrays, computes weights,
        and initializes stimuli, trackers, and commands.
        """
        self.u = np.zeros_like(self.cardiac_tissue.mesh, dtype=self.npfloat)
        self.u_new = self.u.copy()
        self.step = 0
        self.t = 0

        self.cardiac_tissue.compute_myo_indexes()

        if self.stencil is None:
            self.stencil = self.select_stencil(self.cardiac_tissue)

        self.weights = self.stencil.compute_weights(self, self.cardiac_tissue)
        self.diffusion_kernel = self.stencil.select_diffusion_kernel()

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

    def run(self, initialize=True, num_of_theads=None):
        """
        Runs the simulation loop. Handles stimuli, diffusion, ionic kernel
        updates, and tracking.

        Parameters
        ----------
        initialize : bool, optional
            Whether to (re)initialize the model before running the simulation.
            Default is True.
        """
        if initialize:
            self.initialize()

        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)

        if num_of_theads is not None:
            if num_of_theads > numba.config.NUMBA_NUM_THREADS:
                import warnings
                warnings.warn(
                    f"Selected number of threads ({num_of_theads}) exceeds the available threads ({numba.config.NUMBA_NUM_THREADS}). "
                    f"Using the maximum available threads instead."
                )
            num_of_theads = min(num_of_theads, numba.config.NUMBA_NUM_THREADS)
            numba.set_num_threads(num_of_theads)

        if self.t_max < self.t:
            raise ValueError("t_max must be greater than current t.")

        if self.state_loader:
            self.state_loader.load()

        iters = int(np.ceil((self.t_max - self.t) / self.dt))
        bar_desc = f"Running {self.__class__.__name__}"

        for _ in tqdm(range(iters), total=iters, desc=bar_desc,
                      disable=not self.prog_bar):

            if self.stim_sequence:
                self.stim_sequence.stimulate_next()

            self.run_diffusion_kernel()
            self.run_ionic_kernel()

            if self.tracker_sequence:
                self.tracker_sequence.tracker_next()

            self.t += self.dt
            self.step += 1
            self.u_new, self.u = self.u, self.u_new

            if self.command_sequence:
                self.command_sequence.execute_next()

            if self.state_saver:
                self.state_saver.save()

            if self.check_termination():
                if self.state_saver:
                    self.state_saver.save()
                break

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

    def run_diffusion_kernel(self):
        """
        Executes the diffusion kernel computation using the current parameters
        and tissue weights.
        """
        self.diffusion_kernel(self.u_new, self.u, self.weights,
                              self.cardiac_tissue.myo_indexes)

    @abstractmethod
    def select_stencil(self, cardiac_tissue):
        """
        Selects the appropriate stencil based on the cardiac tissue properties.

        Parameters
        ----------
        cardiac_tissue : CardiacTissue
            The tissue object representing the cardiac tissue.

        Returns
        -------
        Stencil
            The stencil object to use for diffusion computations.
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
