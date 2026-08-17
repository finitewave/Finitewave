from pathlib import Path
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class ActivationTimeTracker(Tracker):
    """
    A class to track and record the activation time of each cell in a 2D
    cardiac tissue model.

    This tracker monitors the membrane potential of each cell and records
    the time at which the potential crosses a certain threshold, indicating
    cell activation.

    Attributes
    ----------
    act_t : np.ndarray
        Array to store the activation time of each cell in the 2D model grid.
    threshold : float
        The membrane potential threshold value that determines cell activation.
    file_name : str
        Name of the file where the tracked activation time data will be saved.

    """

    def __init__(self, threshold=-40, file_name="act_time", **kwargs):
        """
        Initializes the ActivationTimeTracker with default parameters.
        """
        super().__init__(**kwargs)
        self.act_t = np.ndarray         # Array to store activation times
        self.threshold = threshold      # Threshold for activation (in mV)
        self.file_name = file_name      # Default file name for saving data

    def initialize(self, simulation):
        """
        Initializes the tracker with the simulation model, setting up
        the activation time array.

        Parameters
        ----------
        model : object
            The cardiac tissue model object that contains the grid (`u`) of
            membrane potentials.
        """
        self.simulation = simulation
        # Initialize activation time array with -1 to indicate unactivated cells
        self.act_t = - np.ones_like(self.simulation.cardiac_model.u)
        self.act_t = self.simulation.backend.wrap_array(self.act_t)
        super().initialize(simulation)

    def _track(self):
        """
        Records the activation time of each cell based on the threshold
        crossing.

        The activation time is recorded as the first instance where
        the membrane potential of a cell crosses the threshold value.
        """
        # Update activation times where they are still -1 and the membrane
        # potential exceeds the threshold
        self.act_t = self.simulation.backend.lib.where(
            (self.act_t < 0) & (self.simulation.cardiac_model.u > self.threshold),
            self.simulation.t, self.act_t)

    @property
    def output(self):
        """
        Returns the tracked activation time data.

        Returns
        -------
        np.ndarray
            The array containing the activation time of each cell in the grid.
        """
        tissue_indexes = self.simulation.cardiac_tissue.tissue_indexes
        output = np.zeros_like(self.simulation.cardiac_tissue.mesh, dtype=np.float64)
        output.flat[tissue_indexes] = np.asarray(self.act_t)
        return output
