from finitewave.core.stimulation.stim_voltage import StimVoltage


class StimVoltageMatrixFDM(StimVoltage):
    """
    A class that applies a voltage stimulus to a 2D cardiac tissue model
    according to a specified matrix.
    """
    def __init__(self, time, volt_value, matrix):
        """
        Initializes the StimVoltageMatrixFDM instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        volt_value : float
            The voltage value to apply.
        matrix : numpy.ndarray
            A 2D array where the voltage stimulus is applied to locations with
            values greater than 0.
        """
        super().__init__(time, volt_value)
        self.matrix = matrix

    def initialize(self, simulation):
        super().initialize(simulation)
        self.simulation = simulation
        self.mask = (self.matrix > 0) & (simulation.cardiac_tissue.mesh == 1)

    def stimulate(self, simulation):
        """
        Applies the voltage stimulus to the cardiac tissue model based on the
        specified matrix.

        The voltage is applied only if the current time is within the
        stimulation period and the stimulation has not been previously applied.

        Parameters
        ----------
        model : CardiacModel
            The 2D cardiac tissue model.

        Notes
        -----
        The voltage value is applied to the positions in the cardiac tissue
        where the corresponding value in ``matrix`` is greater than 0,
        and the ``model.cardiac_tissue.mesh`` value is 1.
        """
        simulation.diffusion_model.u[self.mask] = self.volt_value
