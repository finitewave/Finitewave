import numpy as np

from finitewave.core.tracker.tracker import Tracker


class LocalActivationTime2DTracker(Tracker):
    """
    A class to compute and track multiple activation times in a 2D cardiac
    tissue model simulation.

    This tracker monitors the potential across the cardiac tissue and records
    the times when cells surpass a specific threshold, supporting multiple
    activations such as re-entrant waves or multiple excitations.

    The activation times are stored in a array where each element is an array
    storing the activation times for each cell. Arrays can be not fully filled
    if faster cells activate before slower ones. In oreder to get the full
    activation times, the user should select the next closest activation time
    to the desired time base.

    Attributes
    ----------
    act_t : list of np.ndarray
        A list where each element is an array storing activation times for
        each cell. Preferably accessed through the output property.
    threshold : float
        The potential threshold to determine cell activation.
    file_name : str
        The file name for saving the activation times.

    """

    def __init__(self):
        """
        Initializes the MultiActivationTime2DTracker with default parameters.
        """
        Tracker.__init__(self)
        self.act_t = []  # Initialize activation times as an empty array
        self.threshold = -40  # Activation threshold
        self.file_name = "local_act_time_2d"  # Output file name
        self._activated = np.ndarray  # Array to store the activation state

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model and
        precomputes necessary values.

        Parameters
        ----------
        model : CardiacModel
            The cardiac tissue model object containing the data to be tracked.
        """
        self.model = model
        # Initialize with a single layer of -1 (no activation)
        self.act_t = [-np.ones_like(self.model.u)]
        self._activated = np.full(self.model.u.shape, 0, dtype=bool)

    def _track(self):
        """
        Tracks and stores activation times for each cell in
        the model at each time step.
        """
        cross_mask = self.cross_threshold()
        # Check if there are already activated cells in the current
        # activation layer
        if np.any(self.act_t[-1][cross_mask] > -1):
            self.act_t.append(-np.ones(self.model.u.shape))
        # Update activation times where the threshold is crossed
        self.act_t[-1] = np.where(cross_mask, self.model.t, self.act_t[-1])

    def cross_threshold(self):
        """
        Detects the cells that crossed the threshold and are not activated yet.

        Returns
        -------
        np.ndarray
            A binary array where 1 indicates cells that crossed the threshold
            and are not activated yet.
        """
        # Mask for cells that crossed the threshold and are not activated yet
        cross_mask = ((self.model.u >= self.threshold)
                      & (self._activated == 0))
        self._activated = np.where(cross_mask, 1, self._activated)
        # Set inactive cells that backcrossed the threshold after activation
        backcross_mask = ((self.model.u < self.threshold)
                          & (self._activated == 1))
        self._activated = np.where(backcross_mask, 0, self._activated)
        return cross_mask

    @property
    def output(self):
        """
        Returns the activation times.

        Returns
        -------
        np.ndarray
            The array containing the activation times for each cell.
        """
        return np.array(self.act_t)
