import mlx.core as mx
from finitewave.cpuwave.stimul.stim_current_electrodes import (
    StimCurrentElectrodes as StimCurrentElectrodesCPU
)


class StimCurrentElectrodes(StimCurrentElectrodesCPU):
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
    coords : np.ndarray
        The coordinates of the center of the electrodes.
    size : float
        The radius around each coordinate to include in the stimulation.
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
             simulation.dt * self.curr_value
             )
        