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

from .action_potential_tracker_fdm import ActionPotentialTrackerFDM
from .activation_time_tracker_fdm import ActivationTimeTrackerFDM
from .local_activation_time_tracker_fdm import LocalActivationTimeTrackerFDM
from .ecg_tracker_fdm import ECGTrackerFDM
from .multi_variable_tracker_fdm import MultiVariableTrackerFDM
from .variable_tracker_fdm import VariableTrackerFDM
from .period_tracker_fdm import PeriodTrackerFDM
from .spiral_wave_core_tracker_fdm import SpiralWaveCoreTrackerFDM
from .animation_tracker_fdm import AnimationTrackerFDM
