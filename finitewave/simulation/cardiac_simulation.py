from tqdm import tqdm
import numpy as np

from finitewave.core.simulation.cardiac_simulation_base import (
    CardiacSimulationBase
)

from finitewave.numerics.time_integrator.backward_euler_time_integrator import (
    BackwardEulerTimeIntegrator
)
from finitewave.numerics.time_integrator.forward_euler_time_integrator import (
    ForwardEulerTimeIntegrator
)
from finitewave.numerics.fdm.asymmetric_discretization import AsymmetricDiscretization
from finitewave.numerics.fem.finite_element_discretization import FiniteElementDiscretization


class CardiacSimulation(CardiacSimulationBase):
    def initialize(self):
        self.backend = self.select_backend(self.backend_name)

        if self.time_integrator is None:
            self.time_integrator = self.default_time_integrator()

        if self.spatial_discretization is None:
            self.spatial_discretization = self.default_spatial_discretization()

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
        self.time_integrator.run()

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
            from finitewave.numerics.backends.numba_backend import NumbaBackend
            return NumbaBackend()
        
        if backend_name == "mlx":
            from finitewave.numerics.backends.mlx_backend import MlxBackend
            return MlxBackend()
        
        if backend_name == "jax":
            from finitewave.numerics.backends.jax_backend import JAXBackend
            return JAXBackend()
        
        raise ValueError(f"Unsupported backend: {backend_name}")

    def default_time_integrator(self):
        """Selects the default time integrator based on the type of cardiac tissue.
        For grid-based tissues, it uses the Forward Euler method. For element-based
        tissues, it uses the Backward Euler method with Conjugate Gradient solver.

         Returns
         -------
         Solver
             The default solver instance based on the tissue type.
        """
        if self.cardiac_tissue.meta["type"] == "Grid":
            return ForwardEulerTimeIntegrator()

        if self.cardiac_tissue.meta["type"] == "Elements":
            return BackwardEulerTimeIntegrator()

        raise ValueError("Unsupported tissue type")
    
    def default_spatial_discretization(self):
        """Selects the default spatial discretization based on tissue type.

        Returns
        -------
        SpatialDiscretization
            The selected spatial discretization instance.
        """
        if self.cardiac_tissue.meta['type'] == 'Grid':
            return AsymmetricDiscretization()

        if self.cardiac_tissue.meta['type'] == 'Elements':
            return FiniteElementDiscretization()
        
        raise ValueError("Unsupported tissue type")
