import numpy as np
from finitewave.core.stimulation.stim_voltage import StimVoltage


class StimVoltageCoord(StimVoltage):
    """
    A class that applies a voltage stimulus to a 2D cardiac tissue model
    within a specified region of interest.

    Parameters
    ----------
    time : float
        The time at which the stimulation starts.
    volt_value : float
        The voltage value to apply to the region of interest.
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
    """

    def __init__(self, time, volt_value, x_min, x_max, y_min, y_max,
                 z_min=0, z_max=0):
        """
        Initializes the StimVoltageGridCoord instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        volt_value : float
            The voltage value to apply.
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
        """
        super().__init__(time, volt_value)

        self._range = [x_min, x_max, y_min, y_max, z_min, z_max]

    @property
    def x_min(self):
        return self._range[0]

    @property
    def x_max(self):
        return self._range[1]

    @property
    def y_min(self):
        return self._range[2]

    @property
    def y_max(self):
        return self._range[3]

    @property
    def z_min(self):
        return self._range[4]

    @property
    def z_max(self):
        return self._range[5]

    @x_min.setter
    def x_min(self, value):
        self._range[0] = value

    @x_max.setter
    def x_max(self, value):
        self._range[1] = value

    @y_min.setter
    def y_min(self, value):
        self._range[2] = value

    @y_max.setter
    def y_max(self, value):
        self._range[3] = value

    @z_min.setter
    def z_min(self, value):
        self._range[4] = value

    @z_max.setter
    def z_max(self, value):
        self._range[5] = value

    def initialize(self, simulation):
        super().initialize(simulation)

        myo_indexes = simulation.cardiac_model.myo_indexes
        mesh_indexes = simulation.cardiac_model.mesh_indexes
        coords = simulation.cardiac_tissue.coords[mesh_indexes]

        mask = np.zeros(coords.shape[0], dtype=bool)
        mask[myo_indexes] = True
        for i_axis in range(coords.shape[1]):
            mask &= ((coords[:, i_axis] >= self._range[2 * i_axis]) &
                     (coords[:, i_axis] <= self._range[2 * i_axis + 1]))

        self.indices = np.flatnonzero(mask)

    def stimulate(self, simulation):
        """
        Applies the voltage stimulus to the cardiac model within the
        specified region of interest.

        The voltage is applied only if the current time is within the
        stimulation period and the stimulation has not been previously applied.

        Parameters
        ----------
        simulation : object
            The simulation object to which the voltage stimulus is applied.
        """
        simulation.cardiac_model.u.flat[self.indices] = self.volt_value
