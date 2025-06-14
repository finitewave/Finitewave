import numpy as np
from finitewave.core.stimulation.stim_current import StimCurrent


class StimCurrentArea2D(StimCurrent):
    """
    A class that applies a stimulation current to a 2D cardiac tissue model
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
    u_max : float
        The maximum value of the membrane potential.
    """
    def __init__(self, time, curr_value, duration, coords=None, u_max=None):
        """
        Initializes the StimCurrentArea2D instance.

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
        super().__init__(time, curr_value, duration)
        self.coords = coords
        self.u_max = u_max

    def add_stim_point(self, coord, mesh, size=None):
        """
        Adds an stimulation point to the area to be stimulated.

        Parameters
        ----------
        coord : numpy.ndarray
            The coordinates of the stimulation point.
        mesh : numpy.ndarray
            The mesh of the cardiac tissue model.
        size : float, optional
            The size of the area to be stimulated. Default is None.
        """
        self._coord = coord

        if size is None:
            self.coords = np.atleast_2d(coord)
            return

        tissue_points = np.argwhere(mesh == 1)
        dist = np.linalg.norm(tissue_points - coord, axis=1)
        self.coords = tissue_points[dist < size]

    def initialize(self, model):
        mask = (model.cardiac_tissue.mesh[tuple(self.coords.T)] == 1)

        if mask.sum() == 0:
            raise ValueError("The specified area does not have healthy cells.")

        self._coords = self.coords[mask]
        return super().initialize(model)

    def stimulate(self, model):
        """
        Applies the stimulation current to the cardiac tissue model based on
        the specified binary matrix.

        The stimulation is applied only if the current time is within the
        stimulation period and the stimulation has not been previously applied.

        Parameters
        ----------
        model : CardiacModel
            The 2D cardiac tissue model.
        """
        inds = tuple(self._coords.T)
        model.u[inds] += model.dt * self.curr_value

        if self.u_max is not None:
            model.u[inds] = np.where(model.u[inds] > self.u_max, self.u_max,
                                     model.u[inds])
