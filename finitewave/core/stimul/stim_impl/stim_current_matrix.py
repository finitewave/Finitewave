from finitewave.core.stimul.stim_type.stim_current import StimCurrent
from finitewave.core.stimul.stim_area.matrix_area import MatrixArea


class StimCurrentMatrix(StimCurrent):
    """
    A class that applies a current stimulus to a 2D or 3D cardiac tissue model
    within a specified region of interest.

    Attributes
    ----------
    time : float
        The time at which the stimulation starts.
    curr_value : float
        The value of the stimulation current.
    duration : float
        The duration of the stimulation.
    stim_indexes : numpy.ndarray
        The indexes of the cardiac model where the current stimulus is applied.
    matrix : numpy.ndarray
        A 2D or 3D mask where the current stimulus is applied.
    """

    def __init__(self, time, curr_value, duration, matrix):
        """
        Initializes the StimCurrentCoord2D instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        curr_value : float
            The value of the stimulation current.
        duration : float
            The duration of the stimulation.
        matrix : numpy.ndarray
            A 2D or 3D mask where the current stimulus is applied.
        """
        super().__init__(time, curr_value, duration)
        self.stim_area = MatrixArea(matrix)
