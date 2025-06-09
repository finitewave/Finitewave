from finitewave.core.stimulation.stim_voltage import StimVoltage


class StimVoltageListMatrix3D(StimVoltage):
    """
    A class that applies a voltage stimulus to a 3D cardiac tissue model
    according to a specified matrix.

    Parameters
    ----------
    time : float
        The time at which the stimulation starts.
    volt_values : array-like
        The voltage values to apply.
    matrix : numpy.ndarray
        A 3D array where the voltage stimulus is applied to locations with
        values greater than 0.
    step : int
        The step of the voltage values array.
    """
    def __init__(self, time, volt_values,  duration, matrix):
        """
        Initializes the StimVoltageMatrix3D instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        volt_values : float
            The voltage value to apply.
        duration : float
            The duration of the stimulation.
        matrix : numpy.ndarray
            A 3D array where the voltage stimulus is applied to locations with
            values greater than 0.
        """
        super().__init__(time, volt_values, duration)
        self.matrix = matrix
        self.step = 0

    def initialize(self, model):
        """
        Prepares the stimulation for application.

        Parameters
        ----------
        model : object
            The cardiac tissue model to which the voltage stimulus will be
            applied.
        """
        super().initialize(model)

        total_steps = int(self.duration / model.dt) + 1

        if len(self.volt_value) < total_steps:
            message = ("The length of the voltage values array should be " +
                       "greater than the total number of steps.")
            raise ValueError(message)

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
        model.u[mask] = self.volt_value[self.step]
        self.step += 1
