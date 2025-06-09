
from finitewave.cpuwave2D.tracker.activation_time_2d_tracker import (
    ActivationTime2DTracker
)


class ActivationTime3DTracker(ActivationTime2DTracker):
    """
    Class that tracks activation times in 3D.
    """
    def __init__(self):
        """
        Initializes the ActivationTime3DTracker with default parameters.
        """
        super().__init__()
