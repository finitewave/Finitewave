from .stim import Stim


class StimVoltageCollection(Stim):
    """A class that applies a collection of voltage stimuli to a cardiac model.

    Attributes
    ----------
    start_time : float
        The time at which the stimulation starts.
    voltage_list : list of float
        The list of voltage values to apply.
    duration : float
        The duration of the stimulation.
    iteration : int
        The current iteration index for the voltage list.
    stimulation : StimVoltage
        The stimulation object that applies the voltage stimulus.
    """

    def __init__(self, start_time, voltage_list, duration):
        """
        Initializes the StimVoltageCollection instance.

        Parameters
        ----------
        start_time : float
            The time at which the stimulation starts.
        voltage_list : list of float
            The list of voltage values to apply.
        duration : float
            The duration of the stimulation.
        """
        super().__init__(start_time, duration)
        self.voltage_list = voltage_list
        self.iteration = 0
        self.stimulation = None

    def initialize(self, simulation):
        """
        Initializes the stimulation object for the simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation instance.
        """
        if simulation.dt * (len(self.voltage_list) - 1) < self.duration:
            msg = "Not enough voltage values for the given duration and dt."
            raise ValueError(msg)

        return self.stimulation.initialize(simulation)

    def stimulate(self, simulation):
        """
        Applies the voltage stimulus to the cardiac model.

        Parameters
        ----------
        simulation : Simulation
            The simulation object to which the voltage stimulus is applied.
        """
        if self.update_status(simulation):
            return

        self.stimulation.volt_value = self.voltage_list[self.iteration]
        self.stimulation.stimulate(simulation)
        self.iteration += 1
