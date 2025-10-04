from finitewave.core.stimulation.stim_voltage import StimVoltage


class StimVoltageGridCoord(StimVoltage):
    """
    A class that applies a voltage stimulus to a 2D cardiac tissue model
    within a specified region of interest.

    Parameters
    ----------
    time : float
        The time at which the stimulation starts.
    volt_value : float
        The voltage value to apply to the region of interest.
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
    def __init__(self, time, volt_value, x_min=None, x_max=None, y_min=None,
                 y_max=None, z_min=None, z_max=None):
        """
        Initializes the StimVoltageGridCoord instance.

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
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

    def initialize(self, simulation):
        super().initialize(simulation)
        self.slices = (slice(self.x_min, self.x_max),
                       slice(self.y_min, self.y_max))

        if simulation.cardiac_tissue.mesh.ndim == 3:
            self.slices += (slice(self.z_min, self.z_max),)

    def stimulate(self, simulation):
        """
        Applies the voltage stimulus to the cardiac model within the
        specified region of interest.

        The voltage is applied only if the current time is within the
        stimulation period and the stimulation has not been previously applied.

        Parameters
        ----------
        simulation : object
            The simulation object to which the voltage stimulus is applied.
        """

        roi_mesh = simulation.cardiac_tissue.mesh[self.slices]
        mask = (roi_mesh == 1)

        simulation.diffusion_model.u[self.slices][mask] = self.volt_value
