from pathlib import Path
from math import sqrt
import pandas as pd
from numba import njit
from numba.typed import List

from finitewave.core.tracker.tracker import Tracker


class SpiralWaveCore2DTracker(Tracker):
    """
    A class to track spiral wave tips in a 2D cardiac tissue model.

    This tracker identifies and records the positions of spiral wave tips by
    analyzing voltage isoline crossings in the simulated 2D grid over time.

    Attributes
    ----------
    threshold : float
        Voltage threshold value for detecting spiral tips.
    file_name : str
        Name of the file to save the tracked spiral tip data.
    spiral_wave_cores : list of pd.DataFrame
        List of tracked spiral core data.
    """

    def __init__(self):
        """
        Initializes the Spiral2DTracker with default parameters.
        """
        Tracker.__init__(self)
        self.threshold = 0.5
        self.file_name = "spiral_wave_core"
        self.sprial_wave_cores = []

    def initialize(self, model):
        """
        Initialize the tracker with the given cardiac tissue model.

        Parameters
        ----------
        model : object
            The cardiac tissue simulation model containing the grid and
            voltage data.
        """
        self.model = model
        self.u_prev = self.model.u.copy()

    def track_tip_line(self, u, u_new, threshold):
        """
        Track spiral wave tips in a 2D grid by detecting crossings of voltage
        isolines.

        Parameters
        ----------
        u : np.ndarray
            2D array representing the old voltage values.
        u_new : np.ndarray
            2D array representing the new voltage values.
        threshold : float
            Voltage threshold value for detecting spiral tips.

        Returns
        -------
        List
            List of detected spiral tip positions.
        """
        return list(_track_tip_line(u, u_new, threshold))

    def _track(self):
        """
        Track spiral tips at each simulation step by analyzing voltage data.

        The tracker is updated at each simulation step, detecting any spiral
        tips based on the voltage data from the previous and current steps.
        """
        tips = self.track_tip_line(self.u_prev, self.model.u, self.threshold)
        tips = pd.DataFrame(tips, columns=["x", "y"])
        tips["time"] = self.model.t
        tips["step"] = self.model.step
        self.sprial_wave_cores.append(tips)
        self.u_prev = self.model.u.copy()

    def write(self):
        """
        Save the tracked spiral tip data to a file.
        """
        self.output.to_csv(Path(self.path, self.file_name).with_suffix(".csv"))

    @property
    def output(self):
        """
        Get the tracked spiral core data.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the tracked spiral core data.
        """
        validated = [df for df in self.sprial_wave_cores if not df.empty]
        if not validated:
            return pd.DataFrame(columns=["x", "y", "time", "step"])

        return pd.concat(validated, ignore_index=True)


@njit
def _correct_tip_pos(i, j, u, u_new, threshold):
    """
    Correct the position of a detected spiral wave tip.

    This function corrects the position of a detected spiral wave tip by
    solving a system of equations to find the intersection of voltage isolines.

    Parameters
    ----------
    i, j : int
        Grid indices.
    u, u_new : np.ndarray
        2D arrays representing the old and new voltage values.
    threshold : float
        Voltage threshold value for detecting spiral tips.
    """
    # Compute various differences for both old and new voltage values
    AC = u[i, j] - u[i, j+1] + u[i+1, j+1] - u[i+1, j]
    GC = u[i, j+1] - u[i, j]
    BC = u[i+1, j] - u[i, j]
    DC = u[i, j] - threshold

    AD = u_new[i, j] - u_new[i, j+1] + u_new[i+1, j+1] - u_new[i+1, j]
    GD = u_new[i, j+1] - u_new[i, j]
    BD = u_new[i+1, j] - u_new[i, j]
    DD = u_new[i, j] - threshold

    # Compute intermediate values for solving the system of equations
    Q = BC * AD - BD * AC
    R = GC * AD - GD * AC
    S = DC * AD - DD * AC

    QOnR = Q / R
    SOnR = S / R

    T = AC * QOnR
    U = AC * SOnR - BC + GC * QOnR
    V = GC * SOnR - DC

    # Calculate the discriminant for the quadratic formula
    discriminant = U * U - 4. * T * V

    if discriminant < 0:
        return

    # Two possible solutions for (x, y) coordinates
    T2 = 2. * T

    if T2 == 0.:
        return

    xn = (-U - sqrt(discriminant)) / T2
    xp = (-U + sqrt(discriminant)) / T2
    yn = -QOnR * xn - SOnR
    yp = -QOnR * xp - SOnR

    # Ensure the points lie within the valid grid range
    if 0 <= xn <= 1 and 0 <= yn <= 1:
        return [xn, yn]

    if 0 <= xp <= 1 and 0 <= yp <= 1:
        return [xp, yp]

    return


@njit
def _apply_threshold(i, j, u, threshold):
    """
    Apply a voltage threshold to a 2D grid to detect spiral wave tips.

    This function applies a voltage threshold to the 2D grid to detect spiral
    wave tips by identifying grid points where the voltage crosses the
    specified threshold.

    Parameters
    ----------
    i, j : int
        Grid indices.
    u : np.ndarray
        2D array representing the voltage values.
    threshold : float
        Voltage threshold value for detecting spiral tips.

    Returns
    -------
    int
        1 if the voltage crosses the threshold; otherwise, 0.
    """
    if (u[i][j] >= threshold and (u[i + 1][j] < threshold
                                  or u[i][j + 1] < threshold
                                  or u[i + 1][j + 1] < threshold)):
        return 1

    if (u[i][j] < threshold and (u[i + 1][j] >= threshold
                                 or u[i][j + 1] >= threshold
                                 or u[i + 1][j + 1] >= threshold)):
        return 1

    return 0


@njit
def _track_tip_line(u, u_new, threshold):
    """
    Track spiral wave tips in a 2D grid by detecting crossings of voltage
    isolines.

    This function searches for spiral tips in XY planes by detecting where
    the voltage crosses specified thresholds in both the old and new voltage
    values.

    Parameters
    ----------
    u, u_new : np.ndarray
        2D arrays representing the old and new voltage values.
    threshold : float
        Voltage threshold value for detecting spiral tips.

    Returns
    -------
    List
        List of detected spiral tip positions.
    """
    out = List()
    size_i, size_j = u.shape
    delta = 5  # Safety margin to avoid boundary

    for i in range(delta, size_i - delta):
        for j in range(delta, size_j - delta):
            counter = _apply_threshold(i, j, u, threshold)

            if counter == 1:
                counter += _apply_threshold(i, j, u_new, threshold)

            if counter == 2:
                correction = _correct_tip_pos(i, j, u, u_new, threshold)

                if correction is not None:
                    out.append([j + correction[1], i + correction[0]])

    return out
