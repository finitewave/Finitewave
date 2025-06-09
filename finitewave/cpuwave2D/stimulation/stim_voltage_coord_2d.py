from finitewave.core.stimulation.stim_voltage import StimVoltage


class StimVoltageCoord2D(StimVoltage):
    """
    A class that applies a voltage stimulus to a 2D cardiac tissue model
    within a specified region of interest.

    Parameters
    ----------
    time : float
        The time at which the stimulation starts.
    volt_value : float
        The voltage value to apply to the region of interest.
    x1 : int
        The starting x-coordinate of the region of interest.
    x2 : int
        The ending x-coordinate of the region of interest.
    y1 : int
        The starting y-coordinate of the region of interest.
    y2 : int
        The ending y-coordinate of the region of interest.
    """
    def __init__(self, time, volt_value, x1, x2, y1, y2):
        """
        Initializes the StimVoltageCoord2D instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        volt_value : float
            The voltage value to apply.
        x1 : int
            The starting x-coordinate of the region of interest.
        x2 : int
            The ending x-coordinate of the region of interest.
        y1 : int
            The starting y-coordinate of the region of interest.
        y2 : int
            The ending y-coordinate of the region of interest.
        """
        super().__init__(time, volt_value)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def stimulate(self, model):
        """
        Applies the voltage stimulus to the cardiac tissue model within the
        specified region of interest.

        The voltage is applied only if the current time is within the
        stimulation period and the stimulation has not been previously applied.

        Parameters
        ----------
        model : object
            The cardiac tissue model to which the voltage stimulus is applied.
        """

        roi_mesh = model.cardiac_tissue.mesh[self.x1: self.x2,
                                             self.y1: self.y2]
        mask = (roi_mesh == 1)

        model.u[self.x1: self.x2, self.y1: self.y2][mask] = self.volt_value
