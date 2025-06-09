import numpy as np
from scipy import spatial
from skimage import measure


class Velocity2DCalculation:
    """
    Class for calculating the velocity of the wavefront.

    """
    def __init__(self):
        pass

    @staticmethod
    def front_velocity(act_t, dr):
        """
        Computes the front velocity of activation based on the activation
        times.

        Parameters
        ----------
        act_t : numpy.ndarray
            Activation times.
        dr : float
            Spatial resolution.

        Returns
        -------
        numpy.ndarray
            Velocity of the wavefront.
        """
        min_time = np.min(act_t[act_t >= 0])
        max_time = np.max(act_t)

        start_coords = np.argwhere(act_t == min_time)
        current_coords = np.argwhere(act_t == max_time)

        tree = spatial.KDTree(start_coords)
        dist, _ = tree.query(current_coords)

        front_vel = np.zeros(act_t.shape)
        front_vel = dist * dr / (max_time - min_time)
        return front_vel

    @staticmethod
    def velocity_vector(act_t, dr, orientation=False, t_min=None, t_max=None):
        """
        Computes the velocity of the wavefront from a single source based on
        the elliptical shape of the wavefront.

        Parameters
        ----------
        act_t : numpy.ndarray
            2D array of activation times.
        dr : float
            Spatial resolution.
        orientation : bool
            If True, the angle of the major axis of the ellipse is returned.

        Returns
        -------
        tuple
            Tuple of the major and minor components of the velocity.
        """

        if t_min is None:
            t_min = np.min(act_t[act_t >= 0])

        if t_max is None:
            t_max = np.max(act_t)

        mask = (act_t >= t_min) & (act_t <= t_max)

        major, minor, angle = Velocity2DCalculation.calc_ellipse_axes(mask)
        major_velocity = major * dr / (t_max - t_min)
        minor_velocity = minor * dr / (t_max - t_min)

        if orientation:
            return major_velocity, minor_velocity, angle

        return major_velocity, minor_velocity

    @staticmethod
    def calc_ellipse_axes(mask):
        """
        Calculate the major and minor axes of the ellipse that best fits the
        wavefront.

        Parameters
        ----------
        mask : numpy.ndarray
            2D array of the wavefront.

        Returns
        -------
        tuple
            Major and minor axes and the angle of the ellipse.
        """
        props = measure.regionprops(mask.astype(int))
        major = 0.5 * props[0].major_axis_length
        minor = 0.5 * props[0].minor_axis_length
        orientation = props[0].orientation
        return major, minor, orientation
