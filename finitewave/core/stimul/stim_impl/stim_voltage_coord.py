
from finitewave.core.stimul.stim_type.stim_voltage import StimVoltage
from finitewave.core.stimul.stim_area.rectangular_area import RectangularArea


class StimVoltageCoord(StimVoltage):
    """
    A class that applies a voltage stimulus to a 2D or 3D cardiac tissue model
    within a specified region of interest.

    Parameters
    ----------
    time : float
        The time at which the stimulation starts.
    volt_value : float
        The voltage value to apply to the region of interest.
    stim_indexes : numpy.ndarray
        The indexes of the cardiac model where the voltage stimulus is applied.
    x_min : int
        The starting x-coordinate of the region of interest.
    x_max : int
        The ending x-coordinate of the region of interest.
    y_min : int
        The starting y-coordinate of the region of interest.
    y_max : int
        The ending y-coordinate of the region of interest.
    z_min : int, optional
        The starting z-coordinate of the region of interest.
    z_max : int, optional
        The ending z-coordinate of the region of interest.
    """

    def __init__(self, time, volt_value, x_min, x_max, y_min, y_max,
                 z_min=0, z_max=0):
        """
        Initializes the StimVoltageCoord instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        volt_value : float
            The voltage value to apply.
        x_min : int
            The starting x-coordinate of the region of interest.
        x_max : int
            The ending x-coordinate of the region of interest.
        y_min : int
            The starting y-coordinate of the region of interest.
        y_max : int
            The ending y-coordinate of the region of interest.
        z_min : int, optional
            The starting z-coordinate of the region of interest.
        z_max : int, optional
            The ending z-coordinate of the region of interest.
        """
        super().__init__(time, volt_value)
        self.stim_area = RectangularArea(x_min, x_max, y_min, y_max, z_min, z_max)
