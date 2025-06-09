from finitewave.core.stimulation.stim import Stim


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

    def initialize(self, model):
        """
        Prepares the stimulation for application.

        Parameters
        ----------
        model : CardiacModel
            The simulation model to which the current stimulation will be
            applied.
        """
        self.passed = False
