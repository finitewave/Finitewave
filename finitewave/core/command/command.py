from abc import ABC, abstractmethod


class Command(ABC):
    """Base class for a command to be executed during a simulation.

    Attributes
    ----------
    t : float
        The time at which the command should be executed.

    passed : bool
        Flag indicating whether the command has been executed.
    """

    def __init__(self, time=None):
        """
        Initializes a Command instance with the specified execution time.

        Parameters
        ----------
        time : float
            The time at which the command should be executed.
        """
        self.t = time
        self.passed = False

    @abstractmethod
    def execute(self, simulation):
        """
        Abstract method for executing the command. This method should be
        implemented by subclasses to define the specific behavior of the
        command.

        Parameters
        ----------
        simulation : Simulation
            The simulation instance on which the command will be executed.
        """
        pass

    def update_status(self, simulation):
        """
        Marks the command as executed.

        Parameters
        ----------
        simulation : Simulation
            The simulation instance on which the command was executed
        """
        self.passed = simulation.t >= self.t
        return self.passed
