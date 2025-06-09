import numpy as np
from finitewave.core.stimulation.stim_current import StimCurrent


class StimCurrentCoord3D(StimCurrent):
    """
    A class that applies a stimulation current to a rectangular region of a 3D
    cardiac tissue model.

    Parameters
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
    z1 : int
        The z-coordinate of the lower-left corner of the rectangular region.
    z2 : int
        The z-coordinate of the upper-right corner of the rectangular region.
    u_max : float, optional
        The maximum value of the membrane potential. Default is None.
    """

    def __init__(self, time, curr_value, duration, x1, x2, y1, y2, z1, z2,
                 u_max=None):
        """
        Initializes the StimCurrentCoord3D instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        curr_value : float
            The value of the stimulation current.
        duration : float
            The duration of the stimulation.
        x1, x2, y1, y2, z1, z2 : int
            The coordinates of the rectangular region to which the stimulation
            current is applied.
        u_max : float, optional
            The maximum value of the membrane potential. Default is None.
        """
        super().__init__(time, curr_value, duration)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2
        self.u_max = u_max

    def stimulate(self, model):
        """
        Applies the stimulation current to the specified rectangular region of
        the cardiac tissue model.

        Parameters
        ----------
        model : object
            The cardiac tissue model to which the stimulation current is 
            applied.
        """
        roi_mesh = model.cardiac_tissue.mesh[self.x1: self.x2,
                                             self.y1: self.y2,
                                             self.z1: self.z2]
        mask = (roi_mesh == 1)

        model.u[self.x1: self.x2,
                self.y1: self.y2,
                self.z1: self.z2][mask] += model.dt * self.curr_value

        if self.u_max is not None:
            u = model.u[self.x1: self.x2,
                        self.y1: self.y2,
                        self.z1: self.z2][mask]

            model.u[self.x1: self.x2,
                    self.y1: self.y2,
                    self.z1: self.z2][mask] = np.where(u > self.u_max,
                                                       self.u_max, u)
