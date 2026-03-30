import numpy as np

from finitewave.core.tracker.tracker import Tracker


class LocalActivationTimeTracker(Tracker):
    """
    A class to compute and track multiple activation times in a cardiac
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

    def __init__(self, threshold=-40, step=1, start_time=0, end_time=np.inf):
        """
        Initializes the LocalActivationTimeTracker with default parameters.

        Parameters
        ----------
        threshold : float, optional
            The potential threshold to determine cell activation. Default is -40.
        step : float, optional
            The time step for tracking activation times. Default is 1.
        start_time : float, optional
            The time after which the tracker starts recording activation times. Default is 0.
        end_time : float, optional
            The time after which the tracker stops recording activation times. Default is infinity.
        """
        Tracker.__init__(self)
        self.act_t = []
        self.threshold = threshold
        self.step = step
        self.start_time = start_time
        self.end_time = end_time
        self.file_name = "lat"
        self.activated = False

    def initialize(self, simulation):
        """
        Initializes the tracker with the simulation model and precomputes
        necessary values.

        Parameters
        ----------
        model : CardiacModel
            The cardiac tissue model object containing the data to be tracked.
        """
        self.simulation = simulation
        # Initialize with a single layer of -1 (no activation)
        self.act_t = [-np.ones_like(self.simulation.cardiac_model.u)]
        self._activated = np.full(self.simulation.cardiac_model.u.shape, 0, dtype=bool)

    def _track(self):
        """
        Tracks and stores activation times for each cell in the model
        at each time step.
        """
        u = self.simulation.cardiac_model.u

        if not self.activated:
            self._activate_tracker(u)
            return
        
        cross_mask = self.is_crossed_threshold(u)
        self._extend_act_t(cross_mask)
        self._update_act_t(cross_mask, self.simulation.t)

    def _activate_tracker(self, u):
        """
        Activates the tracker by initializing the front and back masks based
        on the initial potential values.

        Parameters
        ----------
        u : np.ndarray
            The initial potential values of the cardiac tissue model.
        """
        self.front = (u >= self.threshold)
        self.back = ~self.front
        self.activated = True

    def _update_act_t(self, cross_mask, t_simulation):
        """
        Updates the activation time array with the current time for nodes
        that crossed the threshold.

        Parameters
        ----------
        cross_mask : np.ndarray
            A binary array indicating which cells crossed the threshold in
            the current time step.
        t_simulation : float
            The current simulation time.
        """
        self.act_t[-1] = np.where(cross_mask, t_simulation, self.act_t[-1])
    
    def _extend_act_t(self, cross_mask):
        """
        Extends the activation time array if necessary.

        Parameters
        ----------
        cross_mask : np.ndarray
            A binary array indicating which cells crossed the threshold in
            the current time step.
        """
        if np.any(self.act_t[-1][cross_mask] > -1):
            self.act_t.append(-np.ones_like(self.act_t[-1]))
    
    def is_crossed_threshold(self, u):
        """
        Detects the cells that activated in this step.

        Parameters
        ----------
        u : np.ndarray
            The current potential values of the cardiac tissue model.

        Returns
        -------
        np.ndarray
            A binary array where 1 indicates cells that crossed the threshold.
        """
        self.front = (u >= self.threshold)
        cross_mask = self.front & self.back
        self.back = ~self.front
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
    
    def activation_map(self, time_min, time_max):
        """
        Returns the activation map that starts from the `time_base`.

        Parameters
        ----------
        time_min : float
            Time base after which the activation map should be computed.
        time_max : float
            Maximum time to consider for the activation map. Activation times
            greater than this value will be set to NaN.

        Returns
        -------
        np.ndarray
            The activation map corresponding to the selected time base.
        """
        lat_array = self.output
        closest_indices = np.argmax(lat_array >= time_min, axis=0)

        lat_map = np.take_along_axis(lat_array, closest_indices[None, ...],
                                     axis=0)[0]
        lat_map[lat_map > time_max] = np.nan
        return lat_map
