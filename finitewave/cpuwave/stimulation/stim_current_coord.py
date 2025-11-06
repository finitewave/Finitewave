import numpy as np
from .stim_coord import StimCoord


class StimCurrentCoord(StimCoord):
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
    x_min : int
        The starting x-coordinate of the region of interest.
    x_max : int
        The ending x-coordinate of the region of interest.
    y_min : int
        The starting y-coordinate of the region of interest.
    y_max : int
        The ending y-coordinate of the region of interest.
    z_min : int, optional
        The starting z-coordinate of the region of interest.
    z_max : int, optional
        The ending z-coordinate of the region of interest.
    u_max : float, optional
        The maximum value of the membrane potential. Default is None.
    """

    def __init__(self, time, curr_value, duration, x_min, x_max, y_min, y_max,
                 z_min=None, z_max=None, u_max=None):
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
        x_min : int
            The starting x-coordinate of the region of interest.
        x_max : int
            The ending x-coordinate of the region of interest.
        y_min : int
            The starting y-coordinate of the region of interest.
        y_max : int
            The ending y-coordinate of the region of interest.
        z_min : int, optional
            The starting z-coordinate of the region of interest.
        z_max : int, optional
            The ending z-coordinate of the region of interest.
        u_max : float, optional
            The maximum value of the membrane potential. Default is None.
        """
        super().__init__(x_min, x_max, y_min, y_max, z_min, z_max)
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
