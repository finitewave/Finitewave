import mlx.core as mx
from finitewave.cpuwave.stimul.stim_voltage_matrix import (
    StimVoltageMatrix as StimVoltageMatrixCPU
)


class StimVoltageMatrix(StimVoltageMatrixCPU):
    """
    A class that applies a voltage stimulus to a cardiac model based on a
    specified matrix.

    Attributes
    ----------
    t : float
        The time at which the stimulation starts.
    volt_value : float
        The voltage value to apply to the region of interest.
    stim_indexes : numpy.ndarray
        The indexes of the cardiac model where the voltage stimulus is applied.
    matrix : numpy.ndarray
        A mask where the voltage stimulus is applied.
    """
    def initialize(self, simulation):
        super().initialize(simulation)
        self.stim_indexes = mx.array(self.stim_indexes, dtype=mx.int32)

    def stimulate(self, simulation):
        """
        Applies the voltage stimulus to the cardiac tissue model based on the
        specified matrix.

        Parameters
        ----------
        simulation : Simulation
            The simulation object to which the voltage stimulus is applied.
        """
        simulation.cardiac_model.u[self.stim_indexes] = self.volt_value
