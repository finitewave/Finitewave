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
    start_time : float
        The time at which tracking will begin. Default is 0.
    end_time : float
        The time at which tracking will end. Default is infinity.
    step : int
        The frequency at which tracking will occur. Default is 1.
    iter_counter : int
        A counter to keep track of the number of iterations for tracking purposes.
    model : CardiacModel
        The simulation model to which the tracker is attached. This allows
        the tracker to access the model's state and data during the simulation.
    """
    def __init__(self, start_time=0, end_time=np.inf, step=1):
        self.start_time = start_time
        self.end_time = end_time
        self.step = step
        self.iter_counter = 0
        self.tracking_counter = 0
        self.model = None

    def initialize(self, simulation):
        """
        Abstract method to be implemented by subclasses for initializing
        the tracker with the simulation model.

        Parameters
        ----------
        simulation : Simulation
            The simulation object to which the tracker will be attached.
        """
        self.simulation = simulation
        n_measurements = int(np.ceil((min(self.end_time, simulation.t_max) - self.start_time) / 
                                     (simulation.dt * self.step)))
        self.tracking_times = - np.ones((n_measurements,), dtype=float)

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
        if (self.simulation.t < self.start_time) or (self.simulation.t > self.end_time):
            return

        if self.iter_counter % self.step != 0:
            self.iter_counter += 1
            return

        self._track()
        self.tracking_times[self.tracking_counter] = self.simulation.t
        self.tracking_counter += 1
        self.iter_counter += 1

    def clone(self):
        """
        Creates a deep copy of the current tracker instance.

        Returns
        -------
        Tracker
            A deep copy of the current Tracker instance.
        """
        return copy.deepcopy(self)

    def write(self, path=".", file_name="tracked_data", dir_name=""):
        """
        Writes the tracked data to a file.
        """
        np.save(Path(path, dir_name, file_name).with_suffix('.npy'), self.output)
