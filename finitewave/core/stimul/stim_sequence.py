import numpy as np


class StimSequence:
    """A sequence of stimuli to be applied to the cardiac model.

    This class manages a list of stimulation objects and applies them to the
    model based on the simulation time. It handles the initialization of
    stimuli, adding and removing stimuli, and applying the next set of stimuli
    in the sequence.

    Attributes
    ----------
    sequence : list
        A list of ``Stim`` objects representing the sequence of stimuli to be
        applied to the model.

    model : CardiacModel, optional
        The cardiac model to which the stimuli will be applied. This is set
        during initialization.
    """

    def __init__(self):
        """
        Initializes the StimSequence object with an empty sequence and no
        associated model.
        """
        self.sequence = []
        self.model = None

    def initialize(self, simulation):
        """
        Prepares each stimulus in the sequence for application.

        This method sets up each stimulus based on the provided simulation
        ensuring that each stimulus is ready to be applied according to its
        specified start time.

        Parameters
        ----------
        simulation : Simulation
            The simulation instance.
        """
        self.simulation = simulation
        for stim in self.sequence:
            stim.initialize(simulation)

    def add_stim(self, stim):
        """
        Adds a stimulus to the sequence.

        Parameters
        ----------
        stim : Stim
            The ``Stim`` object to be added to the sequence.
        """
        self.sequence.append(stim)

    def remove_stim(self):
        """
        Removes all stimuli from the sequence.

        This method clears the sequence, effectively removing all stimuli that
        were previously added.
        """
        self.sequence = []

    def stimulate_next(self):
        """
        Applies the next set of stimuli based on the current time in the model.

        This method checks each stimulus in the sequence to determine if it
        should be applied based on the current simulation time. If a stimulus
        is due to be applied and has not yet been marked as passed, it is
        stimulated and then marked as done.
        """
        for stim in self.sequence:
            if stim.update_status(self.simulation):
                continue

            stim.stimulate(self.simulation)
