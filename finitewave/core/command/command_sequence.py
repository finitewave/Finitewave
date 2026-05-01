

class CommandSequence:
    """Manages a sequence of commands to be executed during a simulation.

    Attributes
    ----------
    sequence : list
        A list of ``Command`` instances representing the sequence of commands
        to be executed.

    simulation : CardiacSimulation
        The simulation instance associated with this command sequence.
    """

    def __init__(self):
        self.sequence = []
        self.simulation = None

    def initialize(self, simulation):
        """
        Initializes the CommandSequence with the specified simulation and resets
        the execution status of all commands.

        Parameters
        ----------
        simulation : CardiacSimulation
            The cardiac simulation instance to be used for command execution.
        """
        self.simulation = simulation
        for command in self.sequence:
            command.passed = False

    def add_command(self, command):
        """
        Adds a ``Command`` instance to the sequence.

        Parameters
        ----------
        command : Command
            The command instance to be added to the sequence.
        """
        self.sequence.append(command)

    def remove_commands(self):
        """
        Clears the sequence of all commands.
        """
        self.sequence = []

    def execute_next(self):
        """
        Executes commands whose time has arrived and which have not been
        executed yet.
        """
        for command in self.sequence:
            if not command.passed and command.update_status(self.simulation):
                command.execute(self.simulation)

