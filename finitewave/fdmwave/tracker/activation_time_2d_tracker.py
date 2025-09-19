from pathlib import Path
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class ActivationTime2DTracker(Tracker):
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

    def __init__(self):
        """
        Initializes the ActivationTime2DTracker with default parameters.
        """
        Tracker.__init__(self)
        self.act_t = np.ndarray         # Array to store activation times
        self.threshold = -40            # Threshold for activation (in mV)
        self.file_name = "act_time_2d"  # Default file name for saving data

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model, setting up
        the activation time array.

        Parameters
        ----------
        model : object
            The cardiac tissue model object that contains the grid (`u`) of
            membrane potentials.
        """
        self.model = model
        # Initialize activation time array with -1 to indicate unactivated cells
        self.act_t = - np.ones_like(self.model.u)

    def _track(self):
        """
        Records the activation time of each cell based on the threshold
        crossing.

        The activation time is recorded as the first instance where
        the membrane potential of a cell crosses the threshold value.
        """
        # Update activation times where they are still -1 and the membrane
        # potential exceeds the threshold
        self.act_t = np.where((self.act_t < 0)
                              & (self.model.u > self.threshold),
                              self.model.t, self.act_t)

    @property
    def output(self):
        """
        Returns the tracked activation time data.

        Returns
        -------
        np.ndarray
            The array containing the activation time of each cell in the grid.
        """
        return self.act_t
