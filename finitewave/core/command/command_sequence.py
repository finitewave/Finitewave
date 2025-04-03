

class CommandSequence:
    """Manages a sequence of commands to be executed during a simulation.

    Attributes
    ----------
    sequence : list
        A list of ``Command`` instances representing the sequence of commands
        to be executed.

    model : CardiacModel
        The cardiac model instance on which commands will be executed.
    """

    def __init__(self):
        self.sequence = []
        self.model = None

    def initialize(self, model):
        """
        Initializes the CommandSequence with the specified model and resets
        the execution status of all commands.

        Parameters
        ----------
        model : CardiacModel
            The cardiac model instance to be used for command execution.
        """
        self.model = model
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
            if not command.passed and command.update_status(self.model):
                command.execute(self.model)

