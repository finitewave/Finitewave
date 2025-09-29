from pathlib import Path
import numpy as np

from .variable_tracker_fdm import VariableTrackerFDM


class ActionPotentialTrackerFDM(VariableTrackerFDM):
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
        Initializes the ActionPotentialTrackerFDM with default parameters.
        """
        super().__init__()
        self.var_name = "u"
        self.file_name = "act_pot"

    @property
    def act_pot(self):
        return self.output
    
    def write(self):
        """
        Saves the tracked variables to disk as NumPy files.
        """
        if not Path(self.path, self.dir_name).exists():
            Path(self.path, self.dir_name).mkdir(parents=True)

        np.save(Path(self.path, self.dir_name,
                     self.file_name).with_suffix('.npy'), self.output)
