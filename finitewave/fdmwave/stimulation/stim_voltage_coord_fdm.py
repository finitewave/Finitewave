from finitewave.core.stimulation.stim_voltage import StimVoltage


class StimVoltageCoordFDM(StimVoltage):
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
    z1 : int, optional
        The starting z-coordinate of the region of interest. Default is None.
    z2 : int, optional
        The ending z-coordinate of the region of interest. Default is None.
    """
    def __init__(self, time, volt_value, x1, x2, y1, y2, z1=None, z2=None):
        """
        Initializes the StimVoltageCoordFDM instance.

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
        z1 : int, optional
            The starting z-coordinate of the region of interest. Default is None.
        z2 : int, optional
            The ending z-coordinate of the region of interest. Default is None.
        """
        super().__init__(time, volt_value)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2

    def initialize(self, model):
        super().initialize(model)
        if (self.z1 is None or self.z2 is None) and model.cardiac_tissue.mesh.ndim == 3:
            raise ValueError("z1 and z2 must be specified for 3D stimulation")
        
        self.slices = (slice(self.x1, self.x2),
                       slice(self.y1, self.y2))
        
        if model.cardiac_tissue.mesh.ndim == 3:
            self.slices += (slice(self.z1, self.z2),)

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

        roi_mesh = model.cardiac_tissue.mesh[self.slices]
        mask = (roi_mesh == 1)

        model.u[self.slices][mask] = self.volt_value
