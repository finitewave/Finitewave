import numpy as np
import mlx.core as mx
from warnings import warn

from finitewave.core.model.cardiac_model_base import CardiacModelBase
from finitewave.mlxwave.model.kernel.ionic_mlx_kernel import IonicMlxGenerator


class CardiacModelMlx(CardiacModelBase):
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
        self.myo_indexes = None
        self.tissue_indexes = None
        self.ionic_kernel_generator = IonicMlxGenerator()
        # self.prepacing_generator = PrepacingNumbaGenerator()
        self._model_func = {}
        self._ionic_step_func = None

    def initialize(self, simulation):
        """
        Initializes the model for simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation object.
        """
        self._allocate_arrays(simulation)
        self.rhs = None
        self.compute_indexes(simulation.cardiac_tissue)
        self._initialize_ionic_kernel()
        self.ionic_kernel_args = self._collect_ionic_kernel_args()
        
    def compute_indexes(self, cardiac_tissue):
        """
        Computes the myocyte and tissue indexes based on the cardiac tissue mesh.

        Parameters
        ----------
        cardiac_tissue : CardiacTissue
            The cardiac tissue object.
        """
        if self.memory_save:
            self.myo_indexes = mx.array(cardiac_tissue.myo_on_tissue_indexes)
            self.tissue_indexes = mx.array(cardiac_tissue.tissue_indexes)
            return

        self.myo_indexes = mx.array(cardiac_tissue.myo_indexes)
        self.tissue_indexes = mx.arange(cardiac_tissue.mesh.size)

    def run(self, dt):
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
        
        # print(dt)
        # print(self.u)
        # print(self.ionic_kernel_arg_names)
        # print(self.ionic_kernel_args)
        
        res = self.ionic_kernel(
            dt,
            self.u,
            *self.ionic_kernel_args,
        )
        mx.eval(res)
        mx.synchronize()
        self.rhs = res[0]
        
        # print(res)
        
        i = 0
        for name in self.state_vars:
            if name == "u":
                continue
            self.__dict__[name] = res[i+1]
            i += 1
        
        self.ionic_kernel_args[:len(res) - 1] = res[1:]
    
    def prepacing(self, stim_prepacing, history=True):
        """
        Prepaces the model using the provided stimulation sequence.
        
        Parameters
        ----------
        stim_prepacing : StimPrepacing
            Object containing the stimulation sequence and parameters.
        history : bool, optional
            Whether to store the pacing history in `self.u_pacing`.
        """
        self._initialize_prepacing_kernel(history)
        prepacing_kernel_args = self._collect_prepacing_kernel_args()

        dt = stim_prepacing.dt
        stim_values = stim_prepacing.stim_sequence
        
        if history:
            self.u_pacing = np.zeros(len(stim_values), dtype=np.float32)
            self.u_pacing[0] = self.init_u
            prepacing_kernel_args = [self.u_pacing] + prepacing_kernel_args

        state_vals = self.prepacing_kernel(stim_values, dt,
                                           self.init_u, *prepacing_kernel_args)

        # update initial conditions with the final state after prepacing
        for var, value in zip(self.state_vars, state_vals):
            setattr(self, f"init_{var}", value)
     
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
    
    def set_variables(self, vars, initial=False):
        """
        Updates the model's initial conditions with the provided values.

        Parameters
        ----------
        vars : dict
            Dictionary of variable names and their new initial values.
        initial : bool, optional
            Whether the provided values are initial conditions (default is False).
            If True, the values will be set to `init_{var}` attributes.
            If False, they will be set to the current state variable arrays.
        """
        prefix = "init_" if initial else ""

        for name, value in vars.items():
            if not hasattr(self, f"{prefix}{name}"):
                raise ValueError(f"Variable '{name}' not found in the model.")
            
            if initial:
                setattr(self, f"{prefix}{name}", value)
                continue

            var_data = getattr(self, f"{prefix}{name}")
            var_data[:] = value

    def _initialize_variables_and_parameters(self, ops):
        """
        Initializes the model's variables and parameters based on the provided ops object.
        
        Parameters
        ----------
        ops : Ops
            Object containing the model's variables and parameters.
        """
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
    
    def _initialize_model_func(self, ops, jit_ops):
        """
        Initializes the model's ionic step function and any additional model functions.
        
        Parameters
        ----------
        ops : Ops
            Object containing the model's functions, including the `ionic_step` function.
        jit_ops : dict
            Dictionary of additional model functions used in the `ionic_kernel`,
            where keys are function names and values are jit-compiled functions.
        """
        self._ionic_step_func = ops.ionic_step

        self._model_func = {}
        for key, func in jit_ops.items():
            self._model_func[key] = func

    def _allocate_arrays(self, simulation):
        """
        Allocates the model's state variable arrays based on the simulation's cardiac tissue mesh.
        
        Parameters
        ----------
        simulation : Simulation
            The simulation object containing the cardiac tissue mesh.
        """
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
            array_val = init_val * np.ones(shape, dtype=simulation.npfloat)
            setattr(self, name, mx.array(array_val))

        # validate parameter fields shapes if they are arrays
        for name in self.default_parameters.keys():
            par = getattr(self, name)
            if isinstance(par, np.ndarray):
                if par.shape != shape:
                    raise ValueError(
                        f"param '{name}' shape {par.shape} != tissue shape {shape}"
                    ) 
    
    def _initialize_ionic_kernel(self):
        """Construct the `ionic_kernel` function for the model using the IonicKernelGenerator."""
        res = self.ionic_kernel_generator.generate_model_kernel(self)
        self.ionic_kernel, self.ionic_kernel_arg_names = res

    def _initialize_prepacing_kernel(self, history):
        """Construct the `prepacing_kernel` function for the model using the
        PrepacingKernelGenerator."""
        self.prepacing_generator.history = history
        res = self.prepacing_generator.generate_model_kernel(self)
        self.prepacing_kernel, self.prepacing_kernel_arg_names = res

    def _collect_ionic_kernel_args(self):
        """Collects the arguments for the `ionic_kernel` function based on the `ionic_kernel_arg_names`."""
        return [getattr(self, name) for name in self.ionic_kernel_arg_names]
    
    def _collect_prepacing_kernel_args(self):
        """Collects the arguments for the `prepacing_kernel` function based on the `prepacing_kernel_arg_names`."""
        prepacing_kernel_args = []
        for name in self.prepacing_kernel_arg_names:
            if name in self.state_vars:
                prepacing_kernel_args.append(getattr(self, f"init_{name}"))
                continue
            
            if name in self.state_pars:
                prepacing_kernel_args.append(getattr(self, name))
                continue

            raise ValueError(f"Prepacing kernel argument {name} not found in state variables or parameters.")
        
        return prepacing_kernel_args
