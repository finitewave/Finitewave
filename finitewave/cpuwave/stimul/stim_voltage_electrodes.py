from finitewave.core.stimul.stim_electrodes import StimElectrodes


class StimVoltageElectrodes(StimElectrodes):
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
        super().__init__(coords, size)
        self.t = time
        self.volt_value = volt_value

    def stimulate(self, simulation):
        """
        Apply the voltage stimulus to the cardiac model.

        Parameters
        ----------
        simulation : Simulation
            The simulation object to which the voltage stimulus is applied.
        """
        simulation.cardiac_model.u.flat[self.stim_indexes] = self.volt_value
