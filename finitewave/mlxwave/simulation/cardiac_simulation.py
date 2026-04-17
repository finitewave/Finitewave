
from finitewave.cpuwave.simulation.cardiac_simulation import (
    CardiacSimulation as CardiacSimulationCPU
)
from finitewave.mlxwave.solver.forward_euler_solver import (
    ForwardEulerSolver
)
from finitewave.mlxwave.solver.crank_nicolson_cg_solver import (
    CrankNicolsonCGSolver
)


class CardiacSimulation(CardiacSimulationCPU):
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
    def __init__(self, dt=None, t_max=None):
        super().__init__(dt, t_max)

    def set_num_of_threads(self, num_of_threads):
        """
        Sets the number of threads for parallel computations. This method is
        for compatibility with CPU-based simulations and does not affect GPU computations.
        """
        pass

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
            return CrankNicolsonCGSolver()

        raise ValueError("Unsupported tissue type")
