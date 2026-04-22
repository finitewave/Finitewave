
from finitewave.core.stimul.stim_type.stim_voltage import StimVoltage
from finitewave.core.stimul.stim_area.stim_electrodes import StimElectrodes


class StimVoltageElectrodes(StimVoltage):
    """
    A class that applies a voltage stimulus to specific electrodes in a
    cardiac model.

    Attributes
    ----------
    coords : numpy.ndarray
        The coordinates of the electrodes where the voltage stimulus is
        applied.
    size : float
        The radius around each coordinate to include in the stimulation.
    stim_indexes : numpy.ndarray
        The indexes of the cardiac model where the voltage stimulus is applied.
    t : float
        The time at which the stimulation applies.
    volt_value : float
        The voltage value to apply at the electrodes.
    """
    def __init__(self, time, volt_value, coords, size):
        super().__init__(time, volt_value)
        self.stim_area = StimElectrodes(coords, size)
