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
        """
        super().__init__(matrix)
        self.t = time
        self.curr_value = curr_value
        self.duration = duration

    def stimulate(self, simulation):
        """
        Applies the stimulation current to the specified region of the cardiac
        tissue model.

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

        simulation.solver.u_new.flat[self.stim_indexes] += (
            simulation.dt * self.curr_value
            )
