from pathlib import Path
import numpy as np

from .variable_tracker import VariableTracker


class ActionPotentialTracker(VariableTracker):
    """
    A class to track and record the action potential of a specific cell in
    a cardiac tissue.

    This tracker monitors the membrane potential of a single or multiple cells at each
    step and stores the data in an array for later analysis or visualization.

    Attributes
    ----------
    node_inds : array-like
        The indices of the cell in the tissue where the action potential is tracked.
    act_pot : np.ndarray
        The values of the tracked action potential at the specified grid point.
    """

    def __init__(self, node_inds=None, start_time=0, end_time=np.inf, step=1):
        """
        Initializes the ActionPotentialTracker with default parameters.

        Parameters
        ----------
        node_inds : array-like
            The indices of the cell in the tissue where the action potential is tracked.
        start_time : float, optional
            The time at which tracking will begin. Default is 0.
        end_time : float, optional
            The time at which tracking will end. Default is infinity.
        step : int, optional
            The frequency at which tracking will occur. Default is 1.
        """
        super().__init__(node_inds, "u", start_time, end_time, step)

    @property
    def act_pot(self):
        return self.output

    def write(self, path=".", file_name="act_pot", dir_name=""):
        """
        Saves the tracked action potential data to a file in NumPy format.

        Parameters
        ----------
        path : str, optional
            The directory path where the tracked data will be saved.
            Default is the current directory.
        file_name : str, optional
            The name of the file where the tracked data will be saved.
            Default is "act_pot".
        dir_name : str, optional
            An optional subdirectory name within the specified path to save the file.
            Default is an empty string (no subdirectory).
        """
        if not Path(path, dir_name).exists():
            Path(path, dir_name).mkdir(parents=True)

        np.save(Path(path, dir_name, file_name).with_suffix('.npy'), self.output)
