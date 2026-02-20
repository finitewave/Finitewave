import copy
import warnings
from tqdm import tqdm
import numpy as np
import numba

from finitewave.core.simulation.cardiac_simulation_base import (
    CardiacSimulationBase
)
from .diffusion.diffusion_model import DiffusionModel
from .solver.forward_euler_solver import ForwardEulerSolver
from .solver.crank_nicolson_cg_solver import CrankNicolsonCGSolver


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
        super().__init__()
        self.diffusion_model = DiffusionModel()
        self.solver = None

    def initialize(self):

        if self.solver is None:
            self.solver = self.default_solver()

        super().initialize()

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
        if initialize:
            self.initialize()

        self.limit_num_of_threads(num_of_threads)

        if self.t_max < self.t:
            raise ValueError("t_max must be greater than current t.")

        if self.state_loader:
            self.state_loader.load()

        iters = int(np.ceil((self.t_max - self.t) / self.dt))
        bar_desc = (f"Running {self.cardiac_model.__class__.__name__}" +
                    f" on {self.cardiac_tissue.meta['shape']}" +
                    f" {self.cardiac_tissue.meta['type']}")

        for _ in tqdm(range(iters), total=iters, desc=bar_desc,
                      disable=not self.prog_bar):

            if self.stim_sequence:
                self.stim_sequence.stimulate_next()

            if self.tracker_sequence:
                self.tracker_sequence.tracker_next()

            self.cardiac_model.run(self.dt)
            self.solver.run()

            self.t += self.dt
            self.step += 1

            if self.command_sequence:
                self.command_sequence.execute_next()

            if self.state_saver:
                self.state_saver.save()

            if self.check_termination():
                if self.state_saver:
                    self.state_saver.save()
                break

    def limit_num_of_threads(self, num_of_threads):
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
        if self.cardiac_tissue.meta["type"] == "Grid":
            return ForwardEulerSolver()

        if self.cardiac_tissue.meta["type"] == "Elements":
            return CrankNicolsonCGSolver()

        raise ValueError("Unsupported tissue type")
