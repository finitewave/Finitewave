
from finitewave.core.stimul.stim_type.stim_voltage import StimVoltage
from finitewave.core.stimul.stim_area.matrix_area import MatrixArea


class StimVoltageMatrix(StimVoltage):
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
        super().__init__(time, volt_value)
        self.stim_area = MatrixArea(matrix)
