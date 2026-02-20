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

from .action_potential_grid_tracker import ActionPotentialGridTracker
from .activation_time_tracker import ActivationTimeTracker
from .variable_grid_tracker import VariableGridTracker
from .multi_variable_grid_tracker import MultiVariableGridTracker
from .ecg_grid_tracker import ECGGridTracker
from .frame_tracker import FrameTracker
