from .multi_variable_tracker_fdm import MultiVariableTrackerFDM


class PrepacingTrackerFDM(MultiVariableTrackerFDM):
    """
    PrepacingTrackerFDM

    PrepacingTrackerFDM collects the variables after pacing time.
    Pacing time is measured from the first activation of the cell.

    Attributes
    ----------
    cell_ind : list or list of lists with two indices
        The indices [i, j] of the cell where the variables are tracked.
        List of lists can be used to track multiple cells.
    pacing_time : float
        The pacing time in seconds.
    voltage_threshold : float
        The voltage threshold in mV.

    Notes
    -----
    PrepacingTrackerFDM can not be used with `start_time`, `end_time`, and
    `step` parameters.
    """
    def __init__(self):
        super().__init__()
        self.cell_ind = None
        self.pacing_time = None
        self.voltage_threshold = 0
        self._first_activation_time = None
        self.dir_name = "prepacing"

    def _track(self):
        self.check_first_activation()

        if self._first_activation_time is not None:
            if self.model.t - self._first_activation_time >= self.pacing_time:
                super()._track()
                self.end_time = self.model.t

    def check_first_activation(self):
        if self._first_activation_time is not None:
            return

        voltage = self.model.u[*self.cell_ind]

        if voltage >= self.voltage_threshold:
            self._first_activation_time = self.model.t
