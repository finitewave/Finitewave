from .stim_matrix import StimMatrix


class StimVoltageMatrix(StimMatrix):
    """
    A class that applies a voltage stimulus to a cardiac model based on a
    specified matrix.
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
        super().__init__(matrix)
        self.t = time
        self.volt_value = volt_value

    def stimulate(self, simulation):
        """
        Applies the voltage stimulus to the cardiac tissue model based on the
        specified matrix.

        Parameters
        ----------
        simulation : Simulation
            The simulation object to which the voltage stimulus is applied.
        """
        simulation.cardiac_model.u.flat[self.stim_indexes] = self.volt_value
        simulation.solver.u_new.flat[self.stim_indexes] = self.volt_value
