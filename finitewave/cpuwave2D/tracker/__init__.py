"""
2D Tracker
----------

This module contains classes for tracking the evolution of the wavefront in 2D.

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

from .action_potential_2d_tracker import ActionPotential2DTracker
from .activation_time_2d_tracker import ActivationTime2DTracker
from .animation_2d_tracker import Animation2DTracker
from .ecg_2d_tracker import ECG2DTracker
from .local_activation_time_2d_tracker import LocalActivationTime2DTracker
from .multi_variable_2d_tracker import MultiVariable2DTracker
from .period_2d_tracker import Period2DTracker
from .period_animation_2d_tracker import PeriodAnimation2DTracker
from .spiral_wave_core_2d_tracker import SpiralWaveCore2DTracker
from .variable_2d_tracker import Variable2DTracker
