from abc import ABC, abstractmethod
import copy
import warnings
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

        self.observers = []
        self._buffs = [] # observer buffers

        self.diffusion_kernel = None
        self.ionic_kernel = None

        self.u = np.ndarray
        self.u_new = np.ndarray
        self.weights = np.ndarray
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

        self._allocate_state_arrays()

        self.compute_weights()
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

    def compute_weights(self):
        """
        Computes the weights for the stencil.
        """
        self.cardiac_tissue.compute_myo_indexes()

        if self.stencil is None:
            self.stencil = self.select_stencil(self.cardiac_tissue)

        self.weights = self.stencil.compute_weights(self, self.cardiac_tissue)

    def run(self, initialize=True, num_of_threads=None):
        """
        Runs the simulation loop. Handles stimuli, diffusion, ionic kernel
        updates, and tracking.

        Parameters
        ----------
        initialize : bool, optional
            Whether to (re)initialize the model before running the simulation.
            Default is True.
        """
        self._check_cfl_condition()
        if initialize:
            self.initialize()

        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)

        if num_of_threads is not None:
            if num_of_threads > numba.config.NUMBA_NUM_THREADS:
                warnings.warn(
                    f"Selected number of threads ({num_of_threads}) exceeds the available threads ({numba.config.NUMBA_NUM_THREADS}). "
                    f"Using the maximum available threads instead."
                )
            num_of_theads = min(num_of_threads, numba.config.NUMBA_NUM_THREADS)
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
    
    def reset_variables_to_defaults(self):
        """
        Resets the state variables to their default initial conditions.
        """
        for name, value in self.default_variables.items():
            setattr(self, f"init_{name}", value)

    def reset_parameters_to_defaults(self):
        """
        Resets the parameters to their default values.
        """
        for name, value in self.default_parameters.items():
            setattr(self, name, value)

    def _check_cfl_condition(self, safety_factor=0.75):
        """
        Checks the CFL stability condition for the explicit diffusion scheme and issues a warning 
        if it is likely to be violated.
        """
        if self.D_model is None:
            return

        if self.D_model <= 0:
            return

        dt_limit = self.dr**2 / (2 * self.cardiac_tissue.dimensions * self.D_model)
        dt_recommended = safety_factor * dt_limit

        if self.dt > dt_limit:
            warnings.warn(
                (
                    "The selected time step may violate the CFL stability "
                    "condition for the explicit diffusion scheme.\n\n"
                    f"Current values:\n"
                    f"  dt = {self.dt}\n"
                    f"  dr = {self.dr}\n"
                    f"  D_max = {self.D_model}\n"
                    f"  dimensions = {self.cardiac_tissue.dimensions}\n\n"
                    f"Estimated stable dt <= {dt_limit:.6g}\n"
                    f"Recommended dt <= {dt_recommended:.6g}\n\n"
                    "Consider decreasing dt, increasing dr, or decreasing D."
                ),
                RuntimeWarning,
                stacklevel=2
            )
        
    def _initialize_variables_and_parameters(self, ops):
        self.default_parameters = ops.get_parameters()
        self.default_variables = ops.get_variables()

        self.state_vars = self.default_variables.keys()
        self.state_pars = list(self.default_parameters.keys())

        # expose parameters as direct attributes (scalar or array)
        self.reset_parameters_to_defaults()

        # expose initial conditions as init_*
        self.reset_variables_to_defaults() 

    def _allocate_state_arrays(self):
        # allocate state arrays
        for name in self.default_variables.keys():
            init_val = getattr(self, f"init_{name}")
            if isinstance(init_val, np.ndarray):
                setattr(self, name, init_val)
            else:
                setattr(self, name, init_val * np.ones_like(self.u, dtype=self.npfloat))
            if name == 'u':
                self.u_new = self.u.copy()

        # validate parameter fields shapes if they are arrays
        tissue_shape = self.cardiac_tissue.mesh.shape
        for name in self.default_parameters.keys():
            par = getattr(self, name)
            if isinstance(par, np.ndarray):
                if par.shape != tissue_shape:
                    raise ValueError(
                        f"param '{name}' shape {par.shape} != tissue shape {tissue_shape}"
                    ) 
    
    def _initialize_kernel(self, kernel, exclude_params=[]):
        gen = kernel()
        self._kernel_args_order = gen.args_order[:]

        # args_order: state vars first, then all parameters (stable order for call site)
        param_names = list(self.default_parameters.keys())
        var_names = list(self.default_variables.keys())

        # Tell generator which names are arrays vs scalars (for indexing decisions)
        for name in var_names:
            gen.arrays.append(name)

        for name in param_names:
            if name in exclude_params: # computed internally
                continue
            par = getattr(self, name)
            if np.isscalar(par):
                gen.scalars.append(name)
            elif isinstance(par, np.ndarray):
                gen.arrays.append(name)

        return gen

    def _form_and_verify_observers(self):
        buffs = []
        for obs in self.observers:
            name = obs["name"] 
            try:
                buffs.append(getattr(self, name))
            except AttributeError as e:
                raise AttributeError(
                    f"Observer buffer '{name}' not found on model. "
                    f"Create it before initialize(), e.g.: model.{name} = np.zeros(...)."
                ) from e
        return buffs
