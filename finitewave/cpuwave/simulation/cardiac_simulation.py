from typing import Literal
import warnings
from tqdm import tqdm
import numpy as np
import numba

from finitewave.core.simulation.cardiac_simulation_base import (
    CardiacSimulationBase
)
from .diffusion.diffusion_model import DiffusionModel
from finitewave.cpuwave.solver.forward_euler_solver import ForwardEulerSolver
from finitewave.cpuwave.solver.crank_nicolson_solver import CrankNicolsonSolver


class CardiacSimulation(CardiacSimulationBase):
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
    solver : Solver
        The solver used for time integration of the reaction-diffusion system.
    diffusion_model : DiffusionModel
        The diffusion model to assemble the diffusion operator for the simulation.
    cardiac_model : CardiacModel
        The cardiac model that defines the ionic currents and state variables.
    dt : float
        Time step for the simulation.
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
    track_solution : bool
        Whether to track the solution at previous time steps for use in trackers.
    """
    def __init__(
            self,
            dt : float | None = None,
            t_max : float | None = None,
            backend : Literal["numpy", "numba", "mlx", "jax"] = "numba",
            array_dtype : str = "float64"):
        """
        Initializes the CardiacSimulation instance.
        
        Parameters
        ----------
        dt : float, optional
            Time step for the simulation. If None, it must be set before running.
        t_max : float, optional
            Maximum time for the simulation. If None, it must be set before running.
        backend : str, optional
            The backend to use for computations. Default is "numba".
        array_dtype : str, optional
            The data type for numerical computations. Default is "float64".
            If backend does not support "float64", it will use "float32".
        """
        super().__init__(dt, t_max, backend, array_dtype)
        self.diffusion_model = DiffusionModel()

    def initialize(self):

        if self.solver is None:
            self.solver = self.default_solver()

        super().initialize()

    def run(self, initialize=True, num_of_threads=None, prog_bar=True):
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

        self.set_num_of_threads(num_of_threads)

        if self.t_max < self.t:
            raise ValueError("t_max must be greater than current t.")

        if self.state_loader:
            self.state_loader.load()

        bar_desc = self._create_bar_desc()
        iters = int(np.floor((self.t_max - self.t) / self.dt))

        for _ in tqdm(range(iters), total=iters, desc=bar_desc, disable=not prog_bar):
            if self.iter_step():
                break
        
        # Last iteration tracking
        if self.tracker_sequence:
            self.tracker_sequence.tracker_next()
    
    def iter_step(self):
        """
        Performs a single iteration of the simulation.
        """
        if self.check_termination():

            if self.state_saver:
                self.state_saver.save()

            return True

        if self.tracker_sequence:
            self.tracker_sequence.tracker_next()

        if self.stim_sequence:
            self.stim_sequence.stimulate_next()

        self.cardiac_model.run()
        self.solver.run()

        self.t += self.dt
        self.step += 1

        if self.state_saver:
            self.state_saver.save()

        if self.command_sequence:
            self.command_sequence.execute_next()

        return False

    def _create_bar_desc(self):
        return (f"Running {self.cardiac_model.__class__.__name__}" +
                f" on {self.cardiac_tissue.meta['shape']}" +
                f" {self.cardiac_tissue.meta['type']}")

    def set_num_of_threads(self, num_of_threads):
        """
        Sets the number of threads for Numba parallel operations.

        Parameters
        ----------
        num_of_threads : int or None
            The number of threads to use for Numba parallel operations. If None,
            it will use the maximum available threads minus one to avoid overloading the system.
        """
        max_num_of_threads = numba.config.NUMBA_NUM_THREADS

        if num_of_threads is None:
            num_of_threads = max(1, max_num_of_threads - 1)

        if num_of_threads > max_num_of_threads:
            warnings.warn(
                f"Selected number of threads ({num_of_threads}) exceeds the available threads ({max_num_of_threads}). "
                f"Using the maximum available threads instead."
            )
            num_of_threads = min(num_of_threads, max_num_of_threads)

        numba.set_num_threads(num_of_threads)

    def default_solver(self):
        """Selects the default solver based on the type of cardiac tissue.
        For grid-based tissues, it uses the Forward Euler method. For element-based
        tissues, it uses the Crank-Nicolson method with Conjugate Gradient solver.

         Returns
         -------
         Solver
             The default solver instance based on the tissue type.
        """
        if self.cardiac_tissue.meta["type"] == "Grid":
            return ForwardEulerSolver()

        if self.cardiac_tissue.meta["type"] == "Elements":
            return CrankNicolsonSolver()

        raise ValueError("Unsupported tissue type")
