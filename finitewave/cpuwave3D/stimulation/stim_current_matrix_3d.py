import numpy as np
from finitewave.cpuwave2D.stimulation.stim_current_matrix_2d import (
    StimCurrentMatrix2D
)


class StimCurrentMatrix3D(StimCurrentMatrix2D):
    """
    A class that applies a stimulation current to a 3D cardiac tissue model
    based on a binary matrix.

    Parameters
    ----------
    time : float
        The time at which the stimulation starts.
    curr_value : float
        The value of the stimulation current.
    duration : float
        The duration of the stimulation.
    matrix : numpy.ndarray
        A 3D binary matrix indicating the region of interest for stimulation. 
        Elements greater than 0 represent regions to be stimulated.
    u_max : float, optional
        The maximum value of the membrane potential. Default is None.
    """

    def __init__(self, time, curr_value, duration, matrix, u_max=None):
        """
        Initializes the StimCurrentMatrix3D instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        curr_value : float
            The value of the stimulation current.
        duration : float
            The duration of the stimulation.
        matrix : numpy.ndarray
            A 3D binary matrix indicating the region of interest for
            stimulation.
        u_max : float, optional
            The maximum value of the membrane potential. Default is None.
        """
        super().__init__(time, curr_value, duration, matrix, u_max)
