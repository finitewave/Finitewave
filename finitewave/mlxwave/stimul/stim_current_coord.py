import mlx.core as mx
from finitewave.cpuwave.stimul.stim_current_coord import (
    StimCurrentCoord as StimCurrentCoordCPU
)


class StimCurrentCoord(StimCurrentCoordCPU):
    """
    A class that applies a current stimulus to a 2D or 3D cardiac tissue model
    within a specified region of interest.

    Attributes
    ----------
    t : float
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
    """
    def initialize(self, simulation):
        super().initialize(simulation)
        self.stim_indexes = mx.array(self.stim_indexes, dtype=mx.int32)

    def stimulate(self, simulation):
        """
        Apply the current stimulus to the cardiac model.

        Parameters
        ----------
        simulation : Simulation
            The simulation object to which the current stimulus is applied.
        """
        simulation.cardiac_model.u[self.stim_indexes] += (
            self.curr_value * simulation.dt
        )
