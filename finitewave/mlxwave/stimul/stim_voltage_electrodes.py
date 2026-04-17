import mlx.core as mx
from finitewave.cpuwave.stimul.stim_voltage_electrodes import (
    StimVoltageElectrodes as StimVoltageElectrodesCPU
)


class StimVoltageElectrodes(StimVoltageElectrodesCPU):
    """
    A class that applies a voltage stimulus to specific electrodes in a
    cardiac model.

    Attributes
    ----------
    t : float
        The time at which the stimulation applies.
    volt_value : float
        The voltage value to apply at the electrodes.
    coords : numpy.ndarray
        The coordinates of the electrodes where the voltage stimulus is
        applied.
    size : float
        The radius around each coordinate to include in the stimulation.
    stim_indexes : numpy.ndarray
        The indexes of the cardiac model where the voltage stimulus is applied.
    """
    def initialize(self, simulation):
        super().initialize(simulation)
        self.stim_indexes = mx.array(self.stim_indexes, dtype=mx.int32)

    def stimulate(self, simulation):
        """
        Apply the voltage stimulus to the cardiac model.

        Parameters
        ----------
        simulation : Simulation
            The simulation object to which the voltage stimulus is applied.
        """
        simulation.cardiac_model.u[self.stim_indexes] = self.volt_value
