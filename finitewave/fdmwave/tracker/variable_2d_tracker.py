from pathlib import Path
import numpy as np

from .multi_variable_2d_tracker import MultiVariable2DTracker


class Variable2DTracker(MultiVariable2DTracker):
    """
    A tracker that records the values of specified variables from a 2D model 
    over time at a given grid point.

    Attributes
    ----------
    cell_ind : list
        The indices [i, j] of the cell where the variable is tracked.
    """
    def __init__(self):
        super().__init__()
        self.cell_ind = [1, 1]

    @property
    def var_name(self):
        """
        The name of the variable to be tracked.
        """
        return self.var_list[0]

    @var_name.setter
    def var_name(self, value):
        self.var_list = [value]

    @property
    def output(self):
        """
        Property to get the tracked variable values.

        Returns
        -------
        np.ndarray
            The values of the tracked variable at the specified grid point.
        """
        return self.vars[self.var_name]

    def write(self):
        """
        Saves the tracked variables to disk as NumPy files.
        """
        if not Path(self.path, self.dir_name).exists():
            Path(self.path, self.dir_name).mkdir(parents=True)

        np.save(Path(self.path, self.dir_name,
                     self.var_name).with_suffix('.npy'), self.output)
