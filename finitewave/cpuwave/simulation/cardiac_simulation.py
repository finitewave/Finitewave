from typing import Literal

from tqdm import tqdm
import numpy as np

from finitewave.core.simulation import (
    Simulation
)

from finitewave.cpuwave.solver.forward_euler_solver import ForwardEulerSolver
from finitewave.cpuwave.solver.crank_nicolson_solver import CrankNicolsonSolver
from finitewave.cpuwave.diffusion.diffusion_model import DiffusionModel
from finitewave.cpuwave.diffusion.diffusion_model_elements import DiffusionModelElements


class CardiacSimulation(Simulation):
    
    def __init__(
        self,
        dt : float | None = None,
        t_max : float | None = None,
        backend : Literal["numba", "mlx", "jax"] = "numba"):
        super().__init__(dt=dt, t_max=t_max, backend=backend)

    def initialize(self):

        if self.solver is None:
            self.solver = self.default_solver()

        if self.diffusion_model is None:
            self.diffusion_model = self.default_diffusion_model()

        super().initialize()

    def select_backend(self, backend_name):
        """
        Selects the computational backend for the simulation.

        Parameters
        ----------
        backend_name : Literal["numba", "mlx", "jax"]
            The name of the backend to use. Supported values are "numba", "mlx", and "jax".

        Raises
        ------
        ValueError
            If an unsupported backend name is provided.
        """
        if backend_name == "numba":
            from finitewave.backends.numba_backend import NumbaBackend
            return NumbaBackend()
        
        if backend_name == "mlx":
            from finitewave.backends.mlx_backend import MlxBackend
            return MlxBackend()
        
        if backend_name == "jax":
            from finitewave.backends.jax_backend import JaxBackend
            return JaxBackend()
        
        raise ValueError(f"Unsupported backend: {backend_name}")

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

        self.backend.config(num_of_threads=num_of_threads)

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
