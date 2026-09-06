"""Spatial cardiac reaction model and backend-kernel integration."""

import numpy as np

from finitewave.core.model.cardiac_model_base import CardiacModelBase
from .single_cell_model import SingleCellModel


class CardiacModel(CardiacModelBase):
    """Backend-independent cardiac reaction model.

    The model loads its equations from a Finitewave model plugin, allocates
    state arrays over tissue points, and generates a backend-specific reaction
    kernel.

    Attributes
    ----------
    myo_indexes : array-like
        Backend array containing flat indexes of active myocytes.
    tissue_indexes : array-like
        Backend array containing flat indexes of all tissue points represented
        by the compact model arrays.
    backend : Backend
        Computational backend selected by the simulation.
    ionic_kernel : callable
        Backend-generated kernel that evaluates the reaction model and updates
        non-voltage state variables.
    kernel_arg_names : list of str
        Names of model values passed to ``ionic_kernel``.
    ionic_kernel_args : list
        Backend-wrapped values passed to ``ionic_kernel``.
    """

    def __init__(self):
        """Initialize model metadata and load the configured model plugin."""
        super().__init__()
        self.myo_indexes = None
        self.tissue_indexes = None
        self.ionic_kernel_generator = None

    def initialize(self, simulation):
        """Allocate state and generate the reaction kernel for a simulation.

        Parameters
        ----------
        simulation : Simulation
            Simulation providing the tissue and computational backend.
        """
        self.simulation = simulation
        self.backend = simulation.backend
        self.wrap_indexes()
        self._allocate_arrays(simulation)
        self._initialize_ionic_kernel()
        self.collect_ionic_kernel_args()

    def run(self):
        """Evaluate the reaction model for one simulation time step."""
        res = self.ionic_kernel(
            self.simulation.dt,
            self.myo_indexes,
            self.rhs,
            self.u,
            *self.ionic_kernel_args,
        )
        self._reset_state_variables(res)

    def sync_backend(self):
        self.backend.sync(self.u, self.rhs, *self.ionic_kernel_args)
    
    def collect_ionic_kernel_args(self):
        """Collect and validate arguments required by ``ionic_kernel``.

        Model arrays are converted to backend arrays. Non-scalar arrays must
        contain one value per represented tissue point.

        Returns
        -------
        list
            Backend-wrapped kernel arguments in ``kernel_arg_names`` order.

        Raises
        ------
        ValueError
            If an argument is uninitialized or has an incompatible size.
        """
        size = len(self.simulation.cardiac_tissue.tissue_indexes)
        kernel_args = []
        for name in self.kernel_arg_names:
            val = getattr(self, name)

            if val is None:
                raise ValueError(f"Ionic kernel argument '{name}' is not initialized.")

            val = self.backend.wrap_array(val)

            if hasattr(val, "__array_namespace__") and val.size > 1 and val.size != size:
                raise ValueError(
                    f"Ionic kernel argument '{name}' has size {val.size} which " +
                    f"does not match tissue size {size}."
                )
            
            self.__dict__[name] = val
            kernel_args.append(val)

        self.ionic_kernel_args = kernel_args
        return kernel_args

    def wrap_indexes(self):
        """Convert tissue and myocyte indexes to backend arrays."""
        tissue = self.simulation.cardiac_tissue
        self.myo_indexes = self.backend.wrap_indexes(tissue.myo_indexes)
        self.tissue_indexes = self.backend.wrap_indexes(tissue.tissue_indexes)

    def output(self, var_name="u", dtype=np.float64):
        """Return a model variable expanded to the full tissue shape.

        Locations outside the represented tissue are filled with ``NaN``.

        Parameters
        ----------
        var_name : str, optional
            Variable to return. Default is ``"u"``.
        dtype : numpy dtype, optional
            Output dtype. Default is ``numpy.float64``.

        Returns
        -------
        np.ndarray
            Full-size array with the same shape as the tissue mesh.

        Raises
        ------
        ValueError
            If the requested variable does not exist.
        """
        if not hasattr(self, f"{var_name}"):
            raise ValueError(f"Variable '{var_name}' not found in the model.")
        

        mesh = self.simulation.cardiac_tissue.mesh
        var_data = np.array(getattr(self, f"{var_name}"), copy=False)

        if mesh.shape == var_data.shape:
            return var_data.astype(dtype)

        if mesh.size == var_data.size:
            return var_data.reshape(mesh.shape).astype(dtype)
        
        tissue_indexes = np.array(self.tissue_indexes, copy=False)
        var_mesh = np.full_like(mesh, np.nan, dtype=dtype)
        var_mesh.flat[tissue_indexes] = var_data.astype(dtype)
        return var_mesh
    
    def __getitem__(self, key):
        """Return a full-size model variable by name."""
        return self.output(var_name=key)
       
    def prepacing(self, stim_prepacing, history=False):
        """Compute initial conditions by pacing a single-cell model.
        
        Parameters
        ----------
        stim_prepacing : StimSingleCell
            Single-cell stimulation containing the time step and current trace.
        history : bool, optional
            If True, store pacing times, stimuli, and voltage history. Default
            is False.
        """

        cell_model = SingleCellModel()
        cell_model.cardiac_model = self
        cell_model.stim_sequence = stim_prepacing
        state_vars = cell_model.run(history)

        if history:
            self.pacing_times = cell_model.times
            self.pacing_stims = cell_model.stim_current
            self.u_pacing = cell_model.u_history

        # update initial conditions with the final state after prepacing
        self.set_state_variables(state_vars)
    
    def update_state_variables(self, vars):
        """Update current state arrays on an initialized model.

        Scalar values are broadcast to the current state shape. Updated values
        are converted to backend arrays and the kernel argument list is rebuilt.

        Parameters
        ----------
        vars : dict
            Mapping from state-variable names to scalar or array values.

        Raises
        ------
        ValueError
            If a variable is unknown or an array has an incompatible shape.
        """

        for name, value in vars.items():

            if not hasattr(self, name):
                raise ValueError(f"Variable '{name}' not found in the model.")

            var_data = getattr(self, name)

            if not hasattr(value, "__array_namespace__") or value.size == 1:
                value = value * np.ones_like(var_data)

            if value.shape != var_data.shape:
                raise ValueError(f"Shape of provided value for variable '{name}' " +
                                 "does not match model variable shape.")

            value = self.simulation.backend.wrap_array(value)

            setattr(self, name, value)
        
        self.collect_ionic_kernel_args()

    def initialize_variables_and_parameters(self):
        """Load defaults exposed by the model operations plugin.

        Parameters become direct attributes, while state defaults are exposed
        as ``init_<name>`` attributes. Runtime state arrays are allocated later
        by :meth:`initialize`.
        """
        self.default_parameters = self.ops.get_parameters()
        self.default_variables = self.ops.get_variables()
        self.D_model = self.ops.get_diffusion_coefficient()["D_model"]

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
            setattr(self, name, None)      

    def _allocate_arrays(self, simulation):
        """Allocate state and parameter arrays for represented tissue points.
        
        Parameters
        ----------
        simulation : Simulation
            Simulation providing the tissue layout and backend.
        """
        size = len(simulation.cardiac_tissue.tissue_indexes)
        shape = (size,)

        self.rhs = self.backend.wrap_array(np.zeros(shape, dtype=np.float64))
        # allocate state arrays
        for name in self.default_variables.keys():
            init_val = getattr(self, f"init_{name}")
            init_arry = init_val * np.ones(shape, dtype=np.float64)
            init_arry = self.backend.wrap_array(init_arry)
            setattr(self, f"{name}", init_arry)

        # validate parameter fields shapes if they are arrays
        for name in self.default_parameters.keys():
            par = getattr(self, name)

            if hasattr(par, '__array_namespace__') and par.size > 1:
                if par.shape != shape:
                    raise ValueError(
                        f"param '{name}' shape {par.shape} != tissue shape  {shape}"
                    )

            setattr(self, name, self.backend.wrap_array(par))
    
    def _initialize_ionic_kernel(self):
        """Generate the backend-specific reaction kernel."""
        self.ionic_kernel, self.kernel_arg_names = (
            self.backend.model_generator.generate_model_kernel(self)
        )
    
    def _reset_state_variables(self, new_values):
        """Store the reaction term and state values returned by the kernel."""
        self.rhs = new_values[0]
        for i, name in enumerate(self.state_vars):
            if name == "u":
                continue
            self.__dict__[name] = new_values[i]
            self.ionic_kernel_args[i-1] = new_values[i]
