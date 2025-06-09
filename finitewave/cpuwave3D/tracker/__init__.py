"""
3D Tracker
==========

This module contains classes for tracking the evolution of the wavefront in 3D.
Most of the classes in this module are similar to the ones in the 2D tracker
module. More information can be found in the documentation for the 2D tracker
module.

The tracker classes can be grouped into the following categories:

* Full field trackers that track the entire field and output the results in
  a single array.
* Point trackers that track the evolution of a specific point(s) in the field.
* Animation trackers that track the evolution of the field over time and save
  the results as frames for creating animations.

Each tracker class has basic attributes such as ``start_time``, ``end_time``,
``step``, ``path``, and ``file_name``.

.. note::

    Note that the ``start_time`` and ``end_time`` is given in time units,
    and the ``step`` is the number of time steps between recordings.
"""

from .action_potential_3d_tracker import ActionPotential3DTracker
from .activation_time_3d_tracker import ActivationTime3DTracker
from .local_activation_time_3d_tracker import LocalActivationTime3DTracker
from .animation_slice_3d_tracker import AnimationSlice3DTracker
from .ecg_3d_tracker import ECG3DTracker
from .period_3d_tracker import Period3DTracker
from .period_animation_3d_tracker import PeriodAnimation3DTracker
from .spiral_wave_core_3d_tracker import SpiralWaveCore3DTracker
from .variable_3d_tracker import Variable3DTracker
from .multi_variable_3d_tracker import MultiVariable3DTracker
from .vtk_frame_3d_tracker import VTKFrame3DTracker
from .animation_3d_tracker import Animation3DTracker
