from .stim_coord import StimCoord


class StimVoltageCoord(StimCoord):
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
        super().__init__(x_min, x_max, y_min, y_max, z_min, z_max)
        self.t = time
        self.volt_value = volt_value

    def stimulate(self, simulation):
        """
        Applies the voltage stimulus to the cardiac model within the
        specified region of interest.

        The voltage is applied only if the current time is within the
        stimulation period and the stimulation has not been previously applied.

        Parameters
        ----------
        simulation : Simulation
            The simulation object to which the voltage stimulus is applied.
        """
        # apply the stimulus to both `u` and `u_new` to ensure consistency
        # in calculation of transmembrane current for ECG and EGM trackers:
        simulation.cardiac_model.u.flat[self.stim_indexes] = self.volt_value
        simulation.solver.u_new.flat[self.stim_indexes] = self.volt_value
