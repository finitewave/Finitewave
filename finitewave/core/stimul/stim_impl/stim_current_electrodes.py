import numpy as np
from finitewave.core.stimul.stim_area.coords_area import CoordsArea
from finitewave.core.stimul.stim_type.stim_current import StimCurrent


class StimCurrentElectrodes(StimCurrent):
    """
    A class that applies a current stimulus to a 2D or 3D cardiac tissue model
    within a specified region of interest.

    Attributes
    ----------
    time : float
        The time at which the stimulation starts.
    curr_value : float
        The value of the stimulation current.
    duration : float
        The duration of the stimulation.
    stim_indexes : numpy.ndarray
        The indexes of the cardiac model where the current stimulus is applied.
    coords : np.ndarray
        The coordinates of the center of the electrodes.
    size : float
        The radius around each coordinate to include in the stimulation.
    """

    def __init__(self, time, curr_value, duration, coords, size):
        """
        Initializes the StimCurrentCoord2D instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        curr_value : float
            The value of the stimulation current.
        duration : float
            The duration of the stimulation.
        coords : np.ndarray
            The coordinates of the center of the electrodes.
        size : float
            The radius around each coordinate to include in the stimulation.
        """
        super().__init__(time, curr_value, duration)
        self.stim_area = CoordsArea(coords, size)
