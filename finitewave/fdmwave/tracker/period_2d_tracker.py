from pathlib import Path
import numpy as np
import pandas as pd
import json
from .local_activation_time_2d_tracker import LocalActivationTime2DTracker


class Period2DTracker(LocalActivationTime2DTracker):
    """
    A class to track activation periods of cells in a 2D cardiac tissue model
    using detectors.

    Attributes
    ----------
    cell_ind : list or list of lists with two indices
        The indices [i, j] of the cell where the variables are tracked.
        List of lists can be used to track multiple cells.
    file_name : str
        The name of the file to save the computed activation periods.
    """

    def __init__(self):
        """
        Initializes the Period2DTracker with default parameters.
        """
        super().__init__()

        self.cell_ind = []
        self.file_name = "period"  # File name for saving tracked data

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model and preallocates
        memory for tracking.

        Parameters
        ----------
        model : object
            The cardiac tissue model object containing the data to be tracked.
        """
        super().initialize(model)
        self.act_t = [-np.ones(len(np.atleast_2d(self.cell_ind)))]

    def _track(self):
        """
        Tracks and stores activation times for each cell in
        the model at each time step.
        """
        cross_mask = self.cross_threshold()
        cross_mask = cross_mask[tuple(np.atleast_2d(self.cell_ind).T)]
        # Check if there are already activated cells in the current
        # activation layer
        if np.any(self.act_t[-1][cross_mask] > -1):
            self.act_t.append(-np.ones(len(np.atleast_2d(self.cell_ind))))
        # Update activation times where the threshold is crossed
        self.act_t[-1] = np.where(cross_mask, self.model.t, self.act_t[-1])

    @property
    def output(self):
        """
        Property to get the computed activation periods.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the computed activation periods.
        """
        lats = np.array(self.act_t)
        lats = pd.DataFrame(lats.T)
        periods = lats.apply(lambda row: np.diff(row[row != -1]), axis=1)
        return periods

    def write(self):
        """
        Saves the computed activation periods to a CSV file.
        """
        periods = self.output
        periods.to_csv(Path(self.path, self.file_name).with_suffix(".csv"))
