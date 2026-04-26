from pathlib import Path
import numpy as np
from .local_activation_time_tracker import LocalActivationTimeTracker


class PeriodTracker(LocalActivationTimeTracker):
    """
    A class to track activation periods of cells in a cardiac tissue model
    using detectors.

    Attributes
    ----------
    cell_ind : list or list of lists with two indices
        The indices [i, j] of the cell where the variables are tracked.
        List of lists can be used to track multiple cells.
    file_name : str
        The name of the file to save the computed activation periods.
    """

    def __init__(self, node_inds=None, threshold=0.5, **kwargs):
        """
        Initializes the PeriodTracker with default parameters.
        """
        super().__init__(threshold=threshold, **kwargs)

        self.node_inds = node_inds
        self.file_name = "period"
        self.activated = False

    def initialize(self, simulation):
        """
        Initializes the tracker with the simulation model and preallocates
        memory for tracking.

        Parameters
        ----------
        model : object
            The cardiac tissue model object containing the data to be tracked.
        """
        super().initialize(simulation)
        self._node_inds = self._flatten_inds(self.simulation.cardiac_tissue.mesh,
                                             self.simulation.cardiac_model.tissue_indexes,
                                             self.node_inds)
        self.act_t = [-self.simulation.backend.lib.ones(len(self._node_inds))]
        self.activated = False

    def _track(self):
        """
        Tracks and stores activation times for each cell in
        the model at each time step.
        """
        u = self.simulation.cardiac_model.u
        u = self.simulation.backend.select_values(u, self._node_inds)

        if not self.activated:
            self._activate_tracker(u)
            return
        
        cross_mask = self.is_crossed_threshold(u)
        self._extend_act_t(cross_mask)
        self._update_act_t(cross_mask, self.simulation.t)

    @property
    def output(self):
        """
        Property to get the computed activation periods.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the computed activation periods.
        """
        act_times = np.asarray(self.act_t).T
        periods = []
        for act_t in act_times:
            act_t = act_t[act_t > -1]
            if len(act_t) < 2:
                periods.append(np.array([]))
                continue

            periods.append(np.diff(act_t))

        return np.array(periods)

    def write(self):
        """
        Saves the computed activation periods to a CSV file.
        """
        periods = self.output
        periods.to_csv(Path(self.path, self.file_name).with_suffix(".csv"))
