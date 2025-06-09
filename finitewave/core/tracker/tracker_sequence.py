class TrackerSequence:
    """Manages a sequence of trackers for a simulation.

    The ``TrackerSequence`` class allows for the management of multiple
    ``Tracker`` instances. It provides methods to initialize trackers, add or
    remove trackers from the sequence, and iterate over the trackers to perform
    their tracking functions.

    Attributes
    ----------
    sequence : list of Tracker
        List containing the trackers in the sequence. The trackers are executed
        in the order they are added.

    model : CardiacModel or None
        The simulation model to which the trackers are attached. It is set
        during initialization.

    """

    def __init__(self):
        """
        Initializes the TrackerSequence with an empty sequence and no model.
        """
        self.sequence = []
        self.model = None

    def initialize(self, model):
        """
        Initializes all trackers in the sequence with the provided simulation
        model.

        Parameters
        ----------
        model : CardiacModel
            The simulation model to which the trackers will be attached.
        """
        self.model = model
        for tracker in self.sequence:
            tracker.initialize(model)

    def add_tracker(self, tracker):
        """
        Adds a new tracker to the end of the sequence.

        Parameters
        ----------
        tracker : Tracker
            The tracker instance to be added to the sequence.
        """
        self.sequence.append(tracker)

    def remove_trackers(self):
        """
        Removes all trackers from the sequence.
        """
        self.sequence = []

    def tracker_next(self):
        """
        Executes the `track` method of each tracker in the sequence.
        """
        for tracker in self.sequence:
            tracker.track()
