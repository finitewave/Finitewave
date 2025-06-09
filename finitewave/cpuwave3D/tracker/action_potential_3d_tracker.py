
from finitewave.cpuwave2D.tracker.action_potential_2d_tracker import (
    ActionPotential2DTracker
)


class ActionPotential3DTracker(ActionPotential2DTracker):
    """
    Class that tracks action potentials in 3D.
    """
    def __init__(self):
        """
        Initializes the ActionPotential3DTracker with default parameters.
        """
        super().__init__()
