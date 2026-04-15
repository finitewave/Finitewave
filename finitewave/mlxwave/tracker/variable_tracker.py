from pathlib import Path
import numpy as np

from .multi_variable_tracker import MultiVariableTracker


class VariableTracker(MultiVariableTracker):
    """
    A tracker that records the values of specified variables from a model 
    over time at a given tissue point.

    Attributes
    ----------
    node_inds : array-like
        The indices of the tissue where the variable is tracked.
    var_name : str
        The name of the variable to be tracked.
    output : np.ndarray
        The values of the tracked variable at the specified grid point.
    """
    def __init__(self, node_inds=None, var_name=None, start_time=0, end_time=np.inf, step=1):
        """
        Initializes the VariableTracker with default parameters.
        
        Parameters
        ----------
        node_inds : array-like
            The indices of the tissue where the variable is tracked.
        var_name : str
            The name of the variable to be tracked.
        start_time : float, optional
            The time at which tracking will begin. Default is 0.
        end_time : float, optional
            The time at which tracking will end. Default is infinity.
        step : int, optional
            The frequency at which tracking will occur. Default is 1.
        """
        super().__init__(node_inds, [var_name], start_time, end_time, step)

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
        return np.squeeze(self.vars_data[self.var_name])
