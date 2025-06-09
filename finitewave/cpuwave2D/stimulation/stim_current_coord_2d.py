import numpy as np
from finitewave.core.stimulation.stim_current import StimCurrent


class StimCurrentCoord2D(StimCurrent):
    """
    A class that applies a stimulation current to a rectangular region of a 2D
    cardiac tissue model.

    Attributes
    ----------
    time : float
        The time at which the stimulation starts.
    curr_value : float
        The value of the stimulation current.
    duration : float
        The duration of the stimulation.
    x1 : int
        The x-coordinate of the lower-left corner of the rectangular region.
    x2 : int
        The x-coordinate of the upper-right corner of the rectangular region.
    y1 : int
        The y-coordinate of the lower-left corner of the rectangular region.
    y2 : int
        The y-coordinate of the upper-right corner of the rectangular region.
    u_max : float, optional
        The maximum value of the membrane potential. Default is None.
    """

    def __init__(self, time, curr_value, duration, x1, x2, y1, y2, u_max=None):
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
        x1 : int
            The x-coordinate of the lower-left corner of the rectangular.
        x2 : int
            The x-coordinate of the upper-right corner of the rectangular.
        y1 : int
            The y-coordinate of the lower-left corner of the rectangular.
        y2 : int
            The y-coordinate of the upper-right corner of the rectangular.
        u_max : float, optional
            The maximum value of the membrane potential. Default is None.
        """
        super().__init__(time, curr_value, duration)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.u_max = u_max

    def stimulate(self, model):
        """
        Applies the stimulation current to the specified rectangular region of
        the cardiac tissue model.

        The stimulation is applied only if the current time is within the 
        stimulation period and the stimulation has not been previously applied.

        Parameters
        ----------
        model : CardiacModel
            The 2D cardiac tissue model.
        """

        roi_mesh = model.cardiac_tissue.mesh[self.x1:self.x2, self.y1:self.y2]
        mask = (roi_mesh == 1)
        model.u[self.x1:self.x2,
                self.y1:self.y2][mask] += model.dt * self.curr_value

        if self.u_max is not None:
            u = model.u[self.x1: self.x2,
                        self.y1: self.y2][mask]

            model.u[self.x1: self.x2,
                    self.y1: self.y2][mask] = np.where(u > self.u_max,
                                                       self.u_max, u)
