import numpy as np
from warnings import warn

from finitewave.core.model.cardiac_model_base import CardiacModelBase

from ._kernel_generators import StepKernelGenerator
from ._kernel_generators import SingleCellKernelGenerator


class CardiacModel(CardiacModelBase):
    """
    Base class for cardiac grid models.

    Attributes
    ----------
    state_vars : list
        List of state variables to be saved and restored.
    memory_save : bool
        Whether to save memory by only storing the state variables at the
        tissue indexes (``mesh > 0``).
    D_model : float
        Model-specific diffusion coefficient.
    u : np.ndarray
        Array representing the action potential (mV) across the tissue.
    rhs : np.ndarray
        Array representing the sum of the ionic currents multiplied by dt.
    myo_indexes : np.ndarray
        Array of myocyte indices corresponding to cardiac model arrays.
        If memory saving is enabled, the indexes correspond to
        ``mesh.flat[tissue_indexes] == 1``.
    tissue_indexes : np.ndarray
        Array of indices corresponding to the full tissue mesh.
        State variables and rhs correspond to mesh.flat[tissue_indexes]
    """

    def __init__(self, memory_save):
        """
        Initializes the CardiacModel instance with default parameters.

        Parameters
        ----------
        memory_save : bool
            Whether to save memory by only storing the state variables at the
            tissue indexes (``mesh > 0``).
        """
        super().__init__()
        self.memory_save = memory_save
        self.state_vars = []
        self.step = 1
        self.counter = 0
        self.ionic_kernel_generator = StepKernelGenerator()
        self.prepacing_generator = SingleCellKernelGenerator("prepacing_kernel")
        self.observers = []
        self.model_func = {}

    def initialize(self, simulation):
        """
        Initializes the model for simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation object.
        """
        self._allocate_arrays(simulation)
        self.rhs = np.zeros_like(self.u)
        self.compute_indexes(simulation.cardiac_tissue)

    def run(self, dt):
        """
        Executes the ionic kernel for the Aliev-Panfilov model.
        """
        self.counter += 1
        if (self.counter - 1) % self.step != 0:
            return

        self.ionic_kernel(
            self.rhs,
            self.u,
            self.myo_indexes,
            dt,
            *self.ionic_kernel_args,
        )
        
    def _prepacing(self, stim_prepacing):
        """Executes the prepacing sequence for the Aliev-Panfilov model."""

        dt = stim_prepacing.dt
        stim_values = stim_prepacing.stim_sequence
        self.u_pacing = np.zeros(len(stim_values), dtype=self.npfloat)

        prepacing_kernel_args = []
        for name in self.prepacing_kernel_arg_names:
            if name in self.state_vars:
                param_val = getattr(self, f"init_{name}")

                if not np.isscalar(param_val):
                    raise ValueError(f"Prepacing kernel does not support array initial conditions. " +
                                     f"Initial condition 'init_{name}' is an array.")
                
                prepacing_kernel_args.append(param_val)
                continue
            
            if name in self.state_pars:
                param_val = getattr(self, name)
                if not np.isscalar(param_val):
                    raise ValueError(f"Prepacing kernel does not support array parameters. " +
                                     f"Parameter '{name}' is an array.")
            
                prepacing_kernel_args.append(getattr(self, name))
                continue

            raise ValueError(f"Prepacing kernel argument {name} not found in state variables or parameters.")

        state_vals = self.prepacing_kernel(self.u_pacing, stim_values, dt,
                                           self.init_u, *prepacing_kernel_args)

        
        # initial conditions
        for var, value in zip(self.state_vars, state_vals):
            setattr(self, f"init_{var}", value)

    def _initialize_variables_and_parameters(self, ops):
        self.default_parameters = ops.get_parameters()
        self.default_variables = ops.get_variables()

        self.state_vars = self.default_variables.keys()
        self.state_pars = list(self.default_parameters.keys())

        # expose parameters as direct attributes (scalar or array)
        for name, value in self.default_parameters.items():
            setattr(self, name, value)

        # expose initial conditions as init_*
        for name, value in self.default_variables.items():
            setattr(self, f"init_{name}", value)

        # declare arrays (optional, for readability/debug)
        for name in self.default_variables.keys():
            setattr(self, name, np.ndarray)      

    def _allocate_arrays(self, simulation):
        shape = simulation.cardiac_tissue.mesh.shape

        if self.memory_save:
            shape = (len(simulation.cardiac_tissue.tissue_indexes), )
        
        if not self.memory_save:
            tissue_fraction = (len(simulation.cardiac_tissue.tissue_indexes) /
                               simulation.cardiac_tissue.mesh.size)
            if tissue_fraction < 0.5:
                warn(f"Tissue fraction is only {tissue_fraction:.2f}. " +
                     "Consider enabling memory saving for better performance.")

        # allocate state arrays
        for name in self.default_variables.keys():
            init_val = getattr(self, f"init_{name}")
            setattr(self, name, init_val * np.ones(shape, dtype=simulation.npfloat))

        # validate parameter fields shapes if they are arrays
        for name in self.default_parameters.keys():
            par = getattr(self, name)
            if isinstance(par, np.ndarray):
                if par.shape != shape:
                    raise ValueError(
                        f"param '{name}' shape {par.shape} != tissue shape {shape}"
                    ) 
    
    def _initialize_ionic_kernel(self, ionic_step, exclude_params=[]):

        state_vars = self.state_vars

        arrays = ["rhs"] + list(state_vars)
        scalars = []
        
        for param in self.state_pars:
            if param in exclude_params:
                continue

            param_val = getattr(self, param)

            if isinstance(param_val, np.ndarray):
                arrays.append(param)

            if np.isscalar(param_val):
                scalars.append(param)

        res = self.ionic_kernel_generator.generate(ionic_step, arrays, scalars,
                                                   state_vars, self.model_func,
                                                   self.observers)
        self.ionic_kernel, self.ionic_kernel_arg_names = res

    def _initialize_prepacing_kernel(self, ionic_step, exclude_params=[]):

        state_vars = self.state_vars

        arrays = ["u_pacing"]
        scalars = []
        
        for param in self.state_pars:
            if param in exclude_params:
                continue

            param_val = getattr(self, param)

            if not np.isscalar(param_val):
                raise ValueError(f"Prepacing kernel does not support array parameters. " +
                                 f"Parameter 'init_{param}' is an array.")
            scalars.append(param)

        for param in self.state_vars:
            if param in exclude_params:
                continue

            param_val = getattr(self, f"init_{param}")

            if not np.isscalar(param_val):
                raise ValueError(f"Prepacing kernel does not support array initial conditions. " +
                                 f"Initial condition 'init_{param}' is an array.")
            scalars.append(param)

        res = self.prepacing_generator.generate(ionic_step, arrays, scalars,
                                                state_vars, self.model_func,
                                                output_args=state_vars)
        self.prepacing_kernel, self.prepacing_kernel_arg_names = res

    def compute_indexes(self, cardiac_tissue):
        """
        Computes the myocyte indexes. If memory saving is enabled, also
        computes the tissue indexes.

        Parameters
        ----------
        cardiac_tissue : CardiacTissue
            The cardiac tissue object.
        """
        if self.memory_save:
            self.myo_indexes = cardiac_tissue.myo_on_tissue_indexes
            self.tissue_indexes = cardiac_tissue.tissue_indexes
            return

        self.myo_indexes = cardiac_tissue.myo_indexes
        self.tissue_indexes = np.arange(cardiac_tissue.mesh.size)
