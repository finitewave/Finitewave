from pathlib import Path
import numpy as np

from .variable_tracker import VariableTracker


class ThresholdTracker(VariableTracker):
    """
    A class to track and record the threshold crossing events of a specific cell in
    a cardiac tissue.

    This tracker monitors the membrane potential of a single or multiple cells at each
    step and stores the data in an array for later analysis or visualization.

    Attributes
    ----------
    node_inds : array-like
        The indices of the cell in the tissue where the threshold crossing events are tracked.
    """

    def __init__(self, node_inds=None, var_name="u", threshold=0.5, **kwargs):
        """
        Initializes the ThresholdTracker with default parameters.

        Parameters
        ----------
        node_inds : array-like
            The indices of the cell in the tissue where the threshold crossing events are tracked.
            If multiple nodes are provided, the the tracker triggers when all nodes cross the threshold.
        var_name : str, optional
            The name of the variable to track for threshold crossing. Default is "u".
        threshold : float, optional
            The threshold value for detecting crossing events. Default is 0.5.
        **kwargs
            Additional keyword arguments for the Tracker base class.
        """
        super().__init__(node_inds=node_inds, var_name=var_name, **kwargs)
        self.threshold = threshold
        self.threshold_crossed = False
        self.front_crossed = False

    def _track(self):
        var_data = self.model.__dict__[self.var_name]
        var_vals = self.simulation.backend.select_values(var_data, self._node_inds)
            
        if not self.front_crossed:
            self.front_crossed = self.simulation.backend.lib.max(var_vals) > self.threshold
            return False
        
        self.threshold_crossed = self.simulation.backend.lib.max(var_vals) < self.threshold

        if self.threshold_crossed:
            self.end_time = self.simulation.t

        return self.threshold_crossed