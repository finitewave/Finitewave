from finitewave.cpuwave2D.stimulation.stim_current_area_2d import (
    StimCurrentArea2D
)


class StimCurrentArea3D(StimCurrentArea2D):
    """
    A class that applies a stimulation current to a 3D cardiac tissue model
    based on a area coords.

    Attributes
    ----------
    time : float
        The time at which the stimulation starts.
    curr_value : float
        The value of the stimulation current.
    duration : float
        The duration of the stimulation.
    coords : numpy.ndarray
        The coordinates of the area to be stimulated.
    """
    def __init__(self, time, curr_value, duration, coords=None, u_max=None):
        """
        Initializes the StimCurrentArea3D instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        curr_value : float
            The value of the stimulation current.
        duration : float
            The duration of the stimulation.
        coords : numpy.ndarray
            The coordinates of the area to be stimulated.
        u_max : float, optional
            The maximum value of the membrane potential. Default is None.
        """
        super().__init__(time, curr_value, duration, coords, u_max)
