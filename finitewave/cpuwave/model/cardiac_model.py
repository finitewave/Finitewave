import numpy as np
from warnings import warn

from finitewave.core.model.cardiac_model_base import CardiacModelBase
from .single_cell_model import SingleCellModel


class CardiacModel(CardiacModelBase):
    """
    Base class for cardiac grid models.

    Attributes
    ----------
    memory_save : bool
        Whether to save memory by only storing the state variables at the
        tissue indexes (``mesh > 0``).
    myo_indexes : np.ndarray
        Array of indices corresponding to the myocytes in the mesh.
        If `memory_saving` is True, the indexes correspond to `mesh.flat[tissue_indexes[myo_indexes]]`.
        Otherwise, they correspond to `mesh.flat[myo_indexes]`.
    tissue_indexes : np.ndarray
        Array of indices corresponding to the tissue points. For consistency, when `memory_save` is False,
        this will be an array of all indexes in the mesh.
    ionic_kernel_generator : KernelGenerator
        Object that generates the multithreaded `ionic_kernel` function for the model.
    prepacing_generator : KernelGenerator
        Object that generates the signle-cell `prepacing_kernel` function for the model.
    """

    def __init__(self):
        """
        Initializes the CardiacModel instance with default parameters.
        """
        super().__init__()
        self.myo_indexes = None
        self.tissue_indexes = None
        self.ionic_kernel_generator = None

    def initialize(self, simulation):
        """
        Initializes the model for simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation object.
        """
        self.simulation = simulation
        self.backend = simulation.backend
        self.wrap_indexes(simulation.cardiac_tissue)
        self._select_ionic_kernel_generator(self.backend)
        self._allocate_arrays(simulation)
        self._initialize_ionic_kernel()
        self.collect_ionic_kernel_args()

    def run(self):
        """
        Executes the ionic kernel for the current time step.

        Parameters
        ----------
        dt : float
            Time step size for the simulation.
        """

        self.iter_counter += 1
        if (self.iter_counter - 1) % self.step != 0:
            return

        res = self.ionic_kernel(
            self.simulation.dt,
            self.myo_indexes,
            self.rhs,
            self.u,
            *self.ionic_kernel_args,
        )
        self._reset_state_variables(res)
    
    def collect_ionic_kernel_args(self):
        """Collects the arguments for the `ionic_kernel` function based on the `ionic_kernel_arg_names`."""
        size = len(self.simulation.cardiac_tissue.tissue_indexes)
        kernel_args = []
        for name in self.kernel_arg_names:
            val = getattr(self, name)

            if val is None:
                raise ValueError(f"Ionic kernel argument '{name}' is not initialized.")

            val = self.backend.wrap(val)

            if hasattr(val, "__array_namespace__") and val.size > 1 and val.size != size:
                raise ValueError(
                    f"Ionic kernel argument '{name}' has size {val.size} which " +
                    f"does not match tissue size {size}."
                )
            
            self.__dict__[name] = val
            kernel_args.append(val)

        self.ionic_kernel_args = kernel_args
        return kernel_args

    def wrap_indexes(self, cardiac_tissue):
        """
        Computes the myocyte and tissue indexes based on the cardiac tissue mesh.

        Parameters
        ----------
        cardiac_tissue : CardiacTissue
            The cardiac tissue object.
        """
        self.myo_indexes = self.backend.wrap_indexes(cardiac_tissue.myo_indexes)
        self.tissue_indexes = self.backend.wrap_indexes(cardiac_tissue.tissue_indexes)

    def output(self, var_name="u", dtype=np.float64):
        """
        Returns the state variable with the tissue shape for output.

        Parameters
        ----------
        var_name : str
            Name of the variable to output. Default is "u" (transmembrane potential).

        Returns
        -------
        np.ndarray (*mesh.shape)
            The state variable array reshaped to the tissue mesh shape,
            with values only at the tissue indexes.
        """
        if not hasattr(self, f"{var_name}"):
            raise ValueError(f"Variable '{var_name}' not found in the model.")
        

        mesh = self.simulation.cardiac_tissue.mesh
        var_data = np.array(getattr(self, f"{var_name}"), copy=False)
        
        tissue_indexes = np.array(self.tissue_indexes, copy=False)

        var_mesh = np.full_like(mesh, np.nan, dtype=dtype)
        var_mesh.flat[tissue_indexes] = var_data.astype(dtype)
        return var_mesh
       
    def prepacing(self, stim_prepacing, history=False):
        """
        Prepaces the model using the provided stimulation sequence.
        
        Parameters
        ----------
        stim_prepacing : StimSingleCell
            Object containing the stimulation sequence and parameters.
        history : bool, optional
            Whether to store the pacing history in `self.u_pacing`.
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
    
    def update_state_variables(self, vars):
        """
        Updates the model's initial conditions with the provided values.
        The arrays will be wrapped to fulfill the backend requirements.

        Parameters
        ----------
        vars : dict
            Dictionary of variable names and their new values to update.
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

            value = self.simulation.backend.wrap(value)

            setattr(self, name, value)
        
        self.collect_ionic_kernel_args()

    def initialize_variables_and_parameters(self):
        """
        Initializes the model's variables and parameters based on the provided ops object.
        
        Parameters
        ----------
        ops : Ops
            Object containing the model's variables and parameters.
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
        """
        Allocates the model's state variable arrays based on the simulation's cardiac tissue mesh.
        
        Parameters
        ----------
        simulation : Simulation
            The simulation object containing the cardiac tissue mesh.
        """
        size = len(simulation.cardiac_tissue.tissue_indexes)
        shape = (size,)

        self.rhs = self.backend.wrap(np.zeros(shape, dtype=np.float64))
        # allocate state arrays
        for name in self.default_variables.keys():
            init_val = getattr(self, f"init_{name}")
            init_arry = init_val * np.ones(shape, dtype=np.float64)
            init_arry = self.backend.wrap(init_arry)
            setattr(self, f"{name}", init_arry)

        # validate parameter fields shapes if they are arrays
        for name in self.default_parameters.keys():
            par = getattr(self, name)

            if hasattr(par, '__array_namespace__') and par.size > 1:
                if par.shape != shape:
                    raise ValueError(
                        f"param '{name}' shape {par.shape} != tissue shape  {shape}"
                    )

            setattr(self, name, self.backend.wrap(par))
    
    def _initialize_ionic_kernel(self):
        """Construct the `ionic_kernel` function for the model using the IonicKernelGenerator."""
        self.ionic_kernel, self.kernel_arg_names = (
            self.ionic_kernel_generator.generate_model_kernel(self)
        )
    
    def _select_ionic_kernel_generator(self, backend):

        if backend.name == "numba":
            from .kernel.ionic_numba_kernel import IonicNumbaKernel
            self.ionic_kernel_generator = IonicNumbaKernel()
            return

        if backend.name == "mlx":
            from .kernel.ionic_mlx_kernel import IonicMlxKernel
            self.ionic_kernel_generator = IonicMlxKernel()
            return
        
        if backend.name == "jax":
            from .kernel.ionic_jax_kernel import IonicJaxKernel
            self.ionic_kernel_generator = IonicJaxKernel()
            return

        raise ValueError(f"Unsupported backend '{backend.name}' for ionic kernel generation.")
    
    def _reset_state_variables(self, new_values):
        """Updates the model's state variables with the new values from the ionic kernel."""
        self.rhs = new_values[0]
        for i, name in enumerate(self.state_vars):
            if name == "u":
                continue
            self.__dict__[name] = new_values[i]
            self.ionic_kernel_args[i-1] = new_values[i]