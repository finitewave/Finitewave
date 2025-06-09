import numpy as np
from scipy import spatial
from skimage import measure

from finitewave.tools.velocity_2d_calculation import Velocity2DCalculation


class Velocity3DCalculation(Velocity2DCalculation):
    """
    Class for calculating the velocity of the wavefront.

    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def velocity_vector(act_t, dr, orientation=False, t_max=None, t_min=None):
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

        res = Velocity3DCalculation.calc_ellipsoid_axes(mask)
        major, medium, minor, theta, phi = res
        major_velocity = major * dr / (t_max - t_min)
        medium_velocity = medium * dr / (t_max - t_min)
        minor_velocity = minor * dr / (t_max - t_min)

        if orientation:
            return major_velocity, medium_velocity, minor_velocity, theta, phi

        return major_velocity, medium_velocity, minor_velocity

    @staticmethod
    def calc_ellipsoid_axes(mask):
        """
        Calculate the major, medium and minor axes of the ellipsoid
        that best fits the wavefront.

        Parameters
        ----------
        mask : numpy.ndarray
            3D array of the wavefront.

        Returns
        -------
        tuple
            Major, medium, minor axes and the angles theta and phi.
        """

        cov_matrix = measure.inertia_tensor(mask.astype(int))
        eigvals, eigvecs = np.linalg.eig(cov_matrix)

        sorted_ids = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_ids]
        eigvecs = eigvecs[:, sorted_ids]

        major = np.sqrt(2.5 * (eigvals[0] + eigvals[1] - eigvals[2]))
        medium = np.sqrt(2.5 * (eigvals[0] - eigvals[1] + eigvals[2]))
        minor = np.sqrt(2.5 * (-eigvals[0] + eigvals[1] + eigvals[2]))

        major_vec = eigvecs[:, 2]
        theta = np.arccos(major_vec[2])
        phi = np.arctan2(major_vec[1], major_vec[0])
        return major, medium, minor, theta, phi
