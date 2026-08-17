from tqdm import tqdm
import numpy as np

from finitewave.core.simulation.cardiac_simulation_base import (
    CardiacSimulationBase
)

from finitewave.numerics.solver.backward_euler_solver import BackwardEulerSolver
from finitewave.numerics.solver.forward_euler_solver import ForwardEulerSolver
from finitewave.numerics.fdm.diffusion_model import DiffusionModel
from finitewave.numerics.fem.diffusion_model_elements import DiffusionModelElements


class CardiacSimulation(CardiacSimulationBase):
    def initialize(self):

        if self.solver is None:
            self.solver = self.default_solver()

        if self.diffusion_model is None:
            self.diffusion_model = self.default_diffusion_model()

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

        self.backend.config(num_of_threads)

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

    def default_solver(self):
        """Selects the default solver based on the type of cardiac tissue.
        For grid-based tissues, it uses the Forward Euler method. For element-based
        tissues, it uses the Backward Euler method with Conjugate Gradient solver.

         Returns
         -------
         Solver
             The default solver instance based on the tissue type.
        """
        if self.cardiac_tissue.meta["type"] == "Grid":
            return ForwardEulerSolver()

        if self.cardiac_tissue.meta["type"] == "Elements":
            return BackwardEulerSolver()

        raise ValueError("Unsupported tissue type")
    
    def default_diffusion_model(self):
        """Selects the default diffusion model based on tissue type.

        Returns
        -------
        DiffusionModel
            The selected diffusion model instance.
        """
        if self.cardiac_tissue.meta['type'] == 'Grid':
            return DiffusionModel()

        if self.cardiac_tissue.meta['type'] == 'Elements':
            return DiffusionModelElements()
        
        raise ValueError("Unsupported tissue type")
