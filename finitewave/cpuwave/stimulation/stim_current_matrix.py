import numpy as np
from .stim_matrix import StimMatrix


class StimCurrentMatrix(StimMatrix):
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
    matrix : numpy.ndarray
        A 2D or 3D mask where the current stimulus is applied.
    u_max : float, optional
        The maximum value of the membrane potential. Default is None.
    """

    def __init__(self, time, curr_value, duration, matrix, u_max=None):
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
        u_max : float, optional
            The maximum value of the membrane potential. Default is None.
        """
        super().__init__(matrix)
        self.t = time
        self.curr_value = curr_value
        self.duration = duration
        self.u_max = u_max

    def stimulate(self, simulation):
        """
        Applies the stimulation current to the specified rectangular region of
        the cardiac tissue model.

        The stimulation is applied only if the current time is within the 
        stimulation period and the stimulation has not been previously applied.

        Parameters
        ----------
        simulation : Simulation
            The simulation object.
        """

        simulation.cardiac_model.u.flat[self.stim_indexes] += (
            simulation.dt * self.curr_value
            )

        if self.u_max is not None:
            u = simulation.cardiac_model.u.flat[self.stim_indexes]

            simulation.cardiac_model.u.flat[self.stim_indexes] = (
                np.where(u > self.u_max, self.u_max, u)
                )
