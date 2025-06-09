from abc import ABC, abstractmethod


class Stim(ABC):
    """Base class for stimulation in cardiac models.

    The ``Stim`` class represents a general stimulation object used in cardiac
    simulations. It provides methods to manage the timing and state of
    stimulation. Subclasses should implement specific stimulation behaviors.

    Attributes
    ----------
    t : float
        The time at which the stimulation is to occur.
    duration : float
        The duration for which the stimulation will be applied.
    passed : bool
        A flag indicating whether the stimulation has been applied.
    """

    def __init__(self, time, duration=0.0):
        """
        Initializes the Stim object with the specified time.

        Parameters
        ----------
        time : float
            The time at which the stimulation is scheduled to occur.
        duration : float, optional
            The duration for which the stimulation will be applied. The default
            value is 0.0, indicating that the stimulation will be applied
            instantaneously.
        """
        self.t = time
        self.duration = duration
        self.passed = False

    @abstractmethod
    def stimulate(self, model):
        """
        Applies the stimulation to the provided model.
        """
        pass

    @abstractmethod
    def initialize(self, model):
        """
        Prepares the stimulation for application.
        """
        pass

    def update_status(self, model):
        """
        Marks the stimulation as completed.
        """
        self.passed = model.t >= (self.t + self.duration)
