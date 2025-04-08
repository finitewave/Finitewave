
from finitewave.cpuwave2D.tracker.local_activation_time_2d_tracker import (
    LocalActivationTime2DTracker
)


class LocalActivationTime3DTracker(LocalActivationTime2DTracker):
    """
    Class that tracks multiple activation times in 3D.
    """
    def __init__(self):
        """
        Initializes the LocalActivationTime3DTracker with default parameters.
        """
        super().__init__()
