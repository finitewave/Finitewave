import copy
import warnings
from tqdm import tqdm
import numpy as np
import numba

from finitewave.gridywave.cpuwave2D.simulation.cardiac_simulation_2d import (
    CardiacSimulation2D
)   
from finitewave.elementalwave.diffusion.diffusion_model_fem import (
    DiffusionModelFEM
)


class CardiacSimulationFEM(CardiacSimulation2D):
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
        self.diffusion_model = DiffusionModelFEM()

    def sync_cardiac_model(self):
        """
        Syncs the cardiac model with the diffusion model.

        Notes
        -----
        If the cardiac model is using memory saving, the cardiac model
        will be updated with the diffusion model values at the tissue indexes.
        Otherwise, the cardiac model will be updated with the diffusion model
        values.
        """
        indexes = self.cardiac_model.myo_indexes
        self.cardiac_model.u[indexes] = self.diffusion_model.u

    def sync_diffusion_model(self):
        indexes = self.cardiac_model.myo_indexes
        self.diffusion_model.u[:] = self.cardiac_model.u[indexes]
