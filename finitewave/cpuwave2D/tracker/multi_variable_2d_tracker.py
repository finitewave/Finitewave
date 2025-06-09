from pathlib import Path
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class MultiVariable2DTracker(Tracker):
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
        Initializes the MultiVariable2DTracker with default parameters.
        """
        Tracker.__init__(self)
        self.var_list = []  # List of variables to be tracked
        self.cell_ind = [1, 1]  # Cell index to track variables
        self.dir_name = "multi_vars"  # Directory to save tracked variables
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

    def _track(self):
        """
        Tracks and stores the values of each specified variable at each time step.

        This method should be called at each time step of the simulation.
        """
        # Track the value of each variable at the specified cell index
        # Make possible to track multiple cells
        cell_ind = tuple(np.atleast_2d(self.cell_ind).T)
        for var_ in self.var_list:
            var_values = self.model.__dict__[var_]
            self.vars[var_].append(var_values[cell_ind])

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
        # Create the output directory if it does not exist
        if not Path(self.path, self.dir_name).is_dir():
            Path(self.path, self.dir_name).mkdir(parents=True)

        # Save each tracked variable to a file
        for var_ in self.var_list:
            np.save(Path(self.path, self.dir_name, f"{var_}.npy"),
                    self.output[var_])
