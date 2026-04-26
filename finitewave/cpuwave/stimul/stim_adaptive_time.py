import numpy as np
from finitewave.core.stimul.stim import Stim


class StimAdaptiveTime(Stim):
    """
    Stimulus protocol that applies a stimulus at adaptive times based on a threshold tracker.
    """

    def __init__(self, stim, threshold_tracker, delay=0):
        """Initialize the adaptive time stimulation protocol.

        Parameters
        ----------
        stim : Stim
            The stimulus to be applied at the adaptive times.
        threshold_tracker : ThresholdTracker
            The tracker that monitors the threshold crossing to determine when to apply the stimulus.
        delay : float, optional
            The delay time after the threshold crossing before applying the stimulus. Default is 0.
        """
        super().__init__(time=np.inf)
        self.stim = stim
        self.stim.t = np.inf
        self.threshold_tracker = threshold_tracker
        self.stim_area = self.stim.stim_area
        self.delay = delay

    def initialize(self, simulation):
        return self.stim.initialize(simulation)

    def update_status(self, simulation):
        if self.threshold_tracker.threshold_crossed and np.isinf(self.stim.t):
            self.stim.t = self.threshold_tracker.end_time + self.delay
        return self.stim.update_status(simulation)
    
    def stimulate(self, simulation):
        return self.stim.stimulate(simulation)
    