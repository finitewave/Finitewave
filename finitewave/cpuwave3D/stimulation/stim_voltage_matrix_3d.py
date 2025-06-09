from finitewave.core.stimulation.stim_voltage import StimVoltage


class StimVoltageMatrix3D(StimVoltage):
    """
    A class that applies a voltage stimulus to a 3D cardiac tissue model
    according to a specified matrix.

    Parameters
    ----------
    time : float
        The time at which the stimulation starts.
    volt_value : float
        The voltage value to apply.
    matrix : numpy.ndarray
        A 3D array where the voltage stimulus is applied to locations with
        values greater than 0.
    """
    def __init__(self, time, volt_value, matrix):
        """
        Initializes the StimVoltageMatrix3D instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        volt_value : float
            The voltage value to apply.
        matrix : numpy.ndarray
            A 3D array where the voltage stimulus is applied to locations with
            values greater than 0.
        """
        super().__init__(time, volt_value)
        self.matrix = matrix

    def stimulate(self, model):
        """
        Applies the voltage stimulus to the cardiac tissue model based on the
        specified matrix.

        Parameters
        ----------
        model : object
            The cardiac tissue model to which the voltage stimulus is applied.
        """
        mask = (self.matrix > 0) & (model.cardiac_tissue.mesh == 1)
        model.u[mask] = self.volt_value
