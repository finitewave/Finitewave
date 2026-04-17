import numpy as np
from .stim import Stim


class StimCoord(Stim):
    """
    Spatial stimulation based on coordinates.

    Attributes:
    ------------
    passed : bool
        Flag indicating if the stimulation has been applied.
    simulation : Simulation
        The simulation instance.
    stim_indexes : np.ndarray
        Indices of the stimulated nodes.
    x_min : float
        Minimum x-coordinate of the stimulation area.
    x_max : float
        Maximum x-coordinate of the stimulation area.
    y_min : float
        Minimum y-coordinate of the stimulation area.
    y_max : float
        Maximum y-coordinate of the stimulation area.
    z_min : float
        Minimum z-coordinate of the stimulation area.
    z_max : float
        Maximum z-coordinate of the stimulation area.
    """

    def __init__(self, x_min, x_max, y_min, y_max, z_min=None, z_max=None):
        """
        Initializes the StimCoord instance.

        Parameters
        ----------
        x_min : int
            The minimum x-coordinate of the region of interest.
        x_max : int
            The maximum x-coordinate of the region of interest.
        y_min : int
            The minimum y-coordinate of the region of interest.
        y_max : int
            The maximum y-coordinate of the region of interest.
        z_min : int, optional
            The starting z-coordinate of the region of interest.
        z_max : int, optional
            The ending z-coordinate of the region of interest.
        """
        super().__init__(time=0.0, duration=0.0)
        self._range = [x_min, x_max, y_min, y_max, z_min, z_max]

    def initialize(self, simulation):
        """
        Initializes the stimulation with the given simulation instance.
        Set passed to False and build the stimulation area.

        Parameters
        ----------
        simulation : Simulation
            The simulation instance.
        """
        self.simulation = simulation
        self.build_stim_area(simulation)

    def build_stim_area(self, simulation):
        """
        Builds the stimulation area based on the simulation instance.

        Parameters
        ----------
        simulation : Simulation
            The simulation instance.
        """
        myo_indexes = simulation.cardiac_model.myo_indexes
        tissue_indexes = simulation.cardiac_model.tissue_indexes
        coords = simulation.cardiac_tissue.coords[tissue_indexes]

        mask = np.zeros(coords.shape[0], dtype=bool)
        mask[myo_indexes] = True
        for i_axis in range(coords.shape[1]):
            mask &= ((coords[:, i_axis] >= self._range[2 * i_axis]) &
                     (coords[:, i_axis] <= self._range[2 * i_axis + 1]))

        self.stim_indexes = np.flatnonzero(mask)

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
