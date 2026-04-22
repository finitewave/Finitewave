from finitewave.core.stimul.stim import Stim


class StimVoltage(Stim):
    """A stimulation class that sets a voltage value in the cardiac model.

    This class represents a specific type of stimulation where a voltage value
    is applied to the model at a specified time. It extends the base ``Stim``
    class and provides functionality for managing the stimulation process,
    including preparing and finalizing the stimulation.

    Attributes
    ----------
    volt_value : float
        The voltage value to be applied during the stimulation.
    """
    def __init__(self, time, volt_value, duration=0.0):
        """
        Initializes the StimVoltage object with the specified time and
        voltage value.

        Parameters
        ----------
        time : float
            The time at which the voltage stimulation is to occur.
        volt_value : float
            The voltage value to be applied during the stimulation.
        duration : float, optional
            The duration for which the voltage will be applied. The default
            value is 0.0, indicating that the voltage will be applied
            instantaneously.
        """
        super().__init__(time, duration)
        self.volt_value = volt_value

    def stimulate(self, simulation):
        """
        Applies the voltage stimulation to the specified region of the cardiac
        tissue model.

        The stimulation is applied only if the current time is within the 
        stimulation period and the stimulation has not been previously applied.

        Parameters
        ----------
        simulation : Simulation
            The simulation object.
        """
        simulation.cardiac_model.u = simulation.backend.set_values(
            simulation.cardiac_model.u,
            self.stim_indexes,
            self.volt_value
        )
