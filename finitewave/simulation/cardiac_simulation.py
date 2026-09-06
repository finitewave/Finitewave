from numbers import Real

from tqdm import tqdm
import numpy as np

from finitewave.core.simulation.simulation import Simulation

from finitewave.numerics.time_integration.backward_euler_time_integration import (
    BackwardEulerTimeIntegration
)
from finitewave.numerics.time_integration.forward_euler_time_integration import (
    ForwardEulerTimeIntegration
)
from finitewave.numerics.fdm.asymmetric_discretization import (
    AsymmetricDiscretization
)
from finitewave.numerics.fem.finite_element_discretization import (
    FiniteElementDiscretization
)


class CardiacSimulation(Simulation):
    """Coordinate initialization and execution of a cardiac simulation."""

    def initialize(self, **backend_options):
        """Initialize the backend and all simulation components.

        Parameters
        ----------
        **backend_options
            Backend-specific configuration such as ``device``, ``float_dtype``,
            or ``num_of_threads``.
        """
        self._validate_configuration()
        self.backend.config(**backend_options)

        if self.time_integration is None:
            self.time_integration = self.default_time_integration()

        if self.spatial_discretization is None:
            self.spatial_discretization = self.default_spatial_discretization()

        super().initialize()

    def run(self, initialize=True, prog_bar=True, **backend_options):
        """Run the simulation loop.

        The loop applies stimuli, evaluates the cardiac reaction model,
        advances time integration, executes commands, and records output.

        Parameters
        ----------
        initialize : bool, optional
            Whether to (re)initialize the model before running the simulation.
            Default is True.
        prog_bar : bool, optional
            Whether to display a progress bar. Default is True.
        **backend_options
            Backend-specific configuration passed to ``backend.config``.
        """
        if initialize:
            self.initialize(**backend_options)
        else:
            self._validate_configuration()
            if self.backend is None:
                raise RuntimeError(
                    "The simulation must be initialized before calling "
                    "run(initialize=False)."
                )
            self.backend.config(**backend_options)

        if self.t_max < self.t:
            raise ValueError("t_max must be greater than or equal to current t.")

        if self.state_loader:
            self.state_loader.load()

        bar_desc = self._create_bar_desc()
        total = self._remaining_steps()

        with tqdm(total=total, desc=bar_desc, disable=not prog_bar) as bar:
            while not self.iter_step():
                bar.update(1)

                # Commands may change t_max while the simulation is running.
                updated_total = bar.n + self._remaining_steps()
                if updated_total != bar.total:
                    bar.total = updated_total
                    bar.refresh()

        # Last iteration tracking
        if self.tracker_sequence:
            self.tracker_sequence.tracker_next()

    def iter_step(self):
        """Perform one simulation iteration.

        Returns
        -------
        bool
            True when no complete time step remains; otherwise, False.
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
        self.time_integration.run()

        self.t += self.dt
        self.iteration += 1

        if self.iteration % self.backend.sync_step == 0:
            self.cardiac_model.sync_backend()

        if self.state_saver:
            self.state_saver.save()

        if self.command_sequence:
            self.command_sequence.execute_next()

        return False

    def _validate_configuration(self):
        """Validate values and components required to initialize or run."""
        if self.cardiac_model is None:
            raise ValueError("cardiac_model must be set before initialization.")

        if self.cardiac_tissue is None:
            raise ValueError("cardiac_tissue must be set before initialization.")

        if (not isinstance(self.dt, Real) or isinstance(self.dt, bool) or
                not np.isfinite(self.dt)):
            raise ValueError("dt must be a finite positive number.")

        if self.dt <= 0:
            raise ValueError("dt must be greater than zero.")

        if (not isinstance(self.t_max, Real) or isinstance(self.t_max, bool) or
                not np.isfinite(self.t_max)):
            raise ValueError("t_max must be a finite non-negative number.")

        if self.t_max < 0:
            raise ValueError("t_max must be greater than or equal to zero.")

    def _create_bar_desc(self):
        return (f"Running {self.cardiac_model.__class__.__name__}" +
                f" on {self.cardiac_tissue.meta['shape']}" +
                f" {self.cardiac_tissue.meta['type']}")

    def select_backend(self, backend_name):
        """Select the computational backend for the simulation.

        Parameters
        ----------
        backend_name : Literal["numba", "mlx", "jax"]
            Name of the backend to use. Supported values are ``"numba"``,
            ``"mlx"``, and ``"jax"``.

        Returns
        -------
        Backend
            Selected backend instance.

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

    def default_time_integration(self):
        """Select the default time integration for the cardiac tissue.

        Grid tissues use Forward Euler. Element tissues use Backward Euler with
        the Conjugate Gradient linear solver.

        Returns
        -------
        TimeIntegration
            The default time-integration instance based on the tissue type.
        """
        if self.cardiac_tissue.meta["type"] == "Grid":
            return ForwardEulerTimeIntegration()

        if self.cardiac_tissue.meta["type"] == "Elements":
            return BackwardEulerTimeIntegration()

        raise ValueError("Unsupported tissue type")
    
    def default_spatial_discretization(self):
        """Select the default spatial discretization for the cardiac tissue.

        Returns
        -------
        SpatialDiscretization
            The selected spatial discretization instance.
        """
        if self.cardiac_tissue.meta["type"] == "Grid":
            return AsymmetricDiscretization()

        if self.cardiac_tissue.meta["type"] == "Elements":
            return FiniteElementDiscretization()
        
        raise ValueError("Unsupported tissue type")
