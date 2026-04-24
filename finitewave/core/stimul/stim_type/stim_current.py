from finitewave.core.stimul.stim import Stim


class StimCurrent(Stim):
    """A stimulation class that applies a current value to the cardiac model.

    This class represents a type of stimulation where a current is applied to
    the model for a specified duration. It extends the base ``Stim`` class and
    includes methods for preparing the stimulation and updating its status
    based on elapsed time.

    Attributes
    ----------
    curr_value : float
        The current value to be applied during the stimulation.
    """

    def __init__(self, time, curr_value, duration):
        """
        Initializes the StimCurrent object with the specified parameters.

        Parameters
        ----------
        time : float
            The time at which the current stimulation is to start.
        curr_value : float
            The current value to be applied during the stimulation.
        duration : float
            The duration for which the current will be applied.
        """
        super().__init__(time, duration)
        self.curr_value = curr_value

    def stimulate(self, simulation):
        """
        Applies the stimulation current to the specified rectangular region of
        the cardiac tissue model.

        The stimulation is applied only if the current time is within the 
        stimulation period and the stimulation has not been previously applied.

        Parameters
        ----------
        simulation : Simulation
            The simulation object.
        """
        simulation.cardiac_model.u = simulation.backend.add_flat_values(
            simulation.cardiac_model.u,
            self.stim_indexes,
            self.curr_value * simulation.dt
        )
