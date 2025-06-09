import pandas as pd
from finitewave.cpuwave2D.tracker.spiral_wave_core_2d_tracker import (
    SpiralWaveCore2DTracker
)


class SpiralWaveCore3DTracker(SpiralWaveCore2DTracker):
    """
    A class to track spiral wave cores in 3D cardiac tissue simulations.

    The tip tracker detects spiral wave cores by analyzing the voltage data
    on each slice (z-axis) of the 3D tissue mesh.

    """
    def __init__(self):
        super().__init__()

    def _track(self):
        """
        Track spiral tips at each simulation step by analyzing voltage data.

        The tracker is updated at each simulation step, detecting any spiral
        tips based on the voltage data from the previous and current steps.
        """
        for k in range(self.model.u.shape[2]):
            u_prev = self.u_prev[:, :, k]
            u = self.model.u[:, :, k]
            tips = self.track_tip_line(u_prev, u, self.threshold)
            tips = pd.DataFrame(tips, columns=["x", "y"])
            tips["z"] = k
            tips["time"] = self.model.t
            tips["step"] = self.model.step
            self.sprial_wave_cores.append(tips)

        self.u_prev = self.model.u.copy()
