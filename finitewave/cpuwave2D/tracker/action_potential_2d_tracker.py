import numpy as np

from finitewave.core.tracker.tracker import Tracker


class ActionPotential2DTracker(Tracker):
    """
    A class to track and record the action potential of a specific cell in
    a 2D cardiac tissue model.

    This tracker monitors the membrane potential of a single cell at each time
    step and stores the data in an array for later analysis or visualization.

    Attributes
    ----------
    act_pot : np.ndarray
        Array to store the action potential values at each time step.
    cell_ind : list or list of lists with two indices
        Coordinates of the cell to be tracked in the 2D model grid.
    file_name : str
        Name of the file where the tracked action potential data will be saved.
    """

    def __init__(self):
        """
        Initializes the ActionPotential2DTracker with default parameters.
        """
        Tracker.__init__(self)
        self.act_pot = []  # Initialize the array to store action potential
        self.cell_ind = [1, 1]       # Default cell indices to track
        self.file_name = "act_pot"   # Default file name for saving data

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model, setting up
        the action potential array.

        Parameters
        ----------
        model : object
            The cardiac tissue model object that contains simulation parameters
            like `t_max` (maximum time) and `dt` (time step).
        """
        self.model = model

    def _track(self):
        """
        Records the action potential (`u`) of the specified cell at the current
        time step.
        """
        # Make possible to track multiple cells
        cell_ind = tuple(np.atleast_2d(self.cell_ind).T)
        self.act_pot.append(self.model.u[cell_ind])

    @property
    def output(self):
        """
        Returns the tracked action potential data.

        Returns
        -------
        np.ndarray
            The array containing the tracked action potential values.
        """
        return np.squeeze(self.act_pot)
