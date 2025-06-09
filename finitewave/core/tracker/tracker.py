from pathlib import Path
from abc import ABC, abstractmethod
import copy

import numpy as np


class Tracker(ABC):
    """Base class for trackers used in simulations.

    This class provides a base implementation for trackers that monitor and
    record various aspects of the simulation. Trackers can be used to gather
    data such as activation times, wave dynamics, or ECG readings.

    Attributes
    ----------
    model : CardiacModel
        The simulation model to which the tracker is attached. This allows
        the tracker to access the model's state and data during the simulation.

    file_name : str
        The name of the file where the tracked data will be saved.
        Default is an empty string.

    path : str
        The directory path where the tracked data will be saved.
        Default is the current directory.

    start_time : float
        The time step at which tracking will begin. Default is 0.

    end_time : float
        The time step at which tracking will end. Default is infinity.

    step : int
        The frequency at which tracking will occur. Default is 1.
    """

    # __metaclass__ = ABCMeta

    def __init__(self):
        self.model = None
        self.file_name = "tracked_data"
        self.path = "."
        self.start_time = 0
        self.end_time = np.inf
        self.step = 1

    @abstractmethod
    def initialize(self, model):
        """
        Abstract method to be implemented by subclasses for initializing
        the tracker with the simulation model.

        Parameters
        ----------
        model : CardiacModel
            The simulation model to which the tracker will be attached.
        """
        pass

    @abstractmethod
    def _track(self):
        """
        Abstract method to be implemented by subclasses for tracking and
        recording data during the simulation.
        """
        pass

    def track(self):
        """
        Tracks and records data during the simulation.

        This method calls the ``_track`` method at the specified tracking
        frequency and within the specified time range.
        """
        if self.start_time > self.model.t or self.model.t > self.end_time:
            return
        # Check if the current time step is within the tracking frequency
        if self.model.step % self.step != 0:
            return

        self._track()

    def clone(self):
        """
        Creates a deep copy of the current tracker instance.

        Returns
        -------
        Tracker
            A deep copy of the current Tracker instance.
        """
        return copy.deepcopy(self)

    def write(self):
        """
        Writes the tracked data to a file.
        """
        np.save(Path(self.path, self.file_name).with_suffix('.npy'),
                self.output)
