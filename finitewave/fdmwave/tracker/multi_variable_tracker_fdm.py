from pathlib import Path
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class MultiVariableTrackerFDM(Tracker):
    """
    A class to track multiple variables at a specific cell in a 2D cardiac
    tissue model simulation.

    This tracker monitors user-defined variables at a specified cell index and
    records their values over time.

    Attributes
    ----------
    var_list : list of str
        A list of variable names to be tracked.
    cell_ind : list or list of lists with two indices
        The indices [i, j] of the cell where the variables are tracked.
        List of lists can be used to track multiple cells.
    dir_name : str
        The directory name where tracked variables are saved.
    vars : dict
        A dictionary where each key is a variable name, and the value is
        an array of its tracked values over time.

    """

    def __init__(self):
        """
        Initializes the MultiVariableTrackerFDM with default parameters.
        """
        Tracker.__init__(self)
        self.var_list = []  # List of variables to be tracked
        self.cell_ind = None
        self.vars = {}  # Dictionary to store tracked variables

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model and precomputes
        necessary values for each variable.

        Parameters
        ----------
        model : object
            The cardiac tissue model object containing the data to be tracked.
        """
        self.vars = {}
        self.model = model
        # Initialize storage for each variable to be tracked
        for var_ in self.var_list:
            if var_ not in self.model.__dict__:
                raise ValueError(f"Variable '{var_}' not found in model.")
            self.vars[var_] = []

        if self.cell_ind is None:
            self._meausure_mask = model.cardiac_tissue.mesh == 1
        else:
            self._meausure_mask = tuple(np.atleast_2d(self.cell_ind).T)

    def _track(self):
        """
        Tracks and stores the values of each specified variable at each time step.

        This method should be called at each time step of the simulation.
        """
        for var_ in self.var_list:
            var_values = self.model.__dict__[var_]
            self.vars[var_].append(var_values[self._meausure_mask])

    @property
    def output(self):
        """
        Returns the tracked variables data.

        Returns
        -------
        dict
            A dictionary where each key is a variable name, and the value is
            an array of its tracked values over time.
        """
        vars = {}
        for var_ in self.var_list:
            vars[var_] = np.squeeze(self.vars[var_])
        return vars

    def write(self):
        """
        Saves the tracked variables to disk as NumPy files.
        """
        if not Path(self.path, self.dir_name).is_dir():
            Path(self.path, self.dir_name).mkdir(parents=True)

        for var_ in self.var_list:
            np.save(Path(self.path, self.dir_name, f"{var_}.npy"),
                    self.output[var_])
