.. _finitewave:

finitewave
===========

Python package for simulating cardiac electrophysiology using
finite-difference methods. It provides tools for modeling and visualizing the
propagation of electrical waves in cardiac tissue, making it ideal for
researchers and engineers in computational biology, bioengineering, and
related fields.

.. list-table::
   :widths: auto
   :align: center

   * - .. image:: images/spiral_wave_fib.gif
          :width: 200px
          :alt: Image 1
     - .. image:: images/spiral_wave_slab.gif
          :width: 267px
          :alt: Image 2
     - .. image:: images/spiral_wave_lv.gif
          :width: 220px
          :alt: Image 3

Installation
============

To install Finitewave, navigate to the root directory of the project and run:

.. code-block:: bash

    $ python -m build
    $ pip install dist/finitewave-<version>.whl


This will install Finitewave as a Python package on your system.

For development purposes, you can install the package in an editable mode,
which allows changes to be immediately reflected without reinstallation:

.. code-block:: bash

    $ pip install -e .


Requirements
------------

Finitewave requires the following dependencies:

+-----------------+---------+--------------------------------------------------+
| Dependency      | Version | Link                                             |
+=================+=========+==================================================+
| ffmpeg-python   | 0.2.0   | https://pypi.org/project/ffmpeg-python/          |
+-----------------+---------+--------------------------------------------------+
| matplotlib      | 3.9.2   | https://pypi.org/project/matplotlib/             |
+-----------------+---------+--------------------------------------------------+
| natsort         | 8.4.0   | https://pypi.org/project/natsort/                |
+-----------------+---------+--------------------------------------------------+
| numba           | 0.60.0  | https://pypi.org/project/numba/                  |
+-----------------+---------+--------------------------------------------------+
| numpy           | 1.26.4  | https://pypi.org/project/numpy/                  |
+-----------------+---------+--------------------------------------------------+
| pandas          | 2.2.3   | https://pypi.org/project/pandas/                 |
+-----------------+---------+--------------------------------------------------+
| pyvista         | 0.44.1  | https://pypi.org/project/pyvista/                |
+-----------------+---------+--------------------------------------------------+
| scikit-image    | 0.24.0  | https://pypi.org/project/scikit-image/           |
+-----------------+---------+--------------------------------------------------+
| scipy           | 1.14.1  | https://pypi.org/project/scipy/                  |
+-----------------+---------+--------------------------------------------------+
| tqdm            | 4.66.5  | https://pypi.org/project/tqdm/                   |
+-----------------+---------+--------------------------------------------------+

Quick start
===================

This quick start guide will walk you through the basic steps of setting up a
simple cardiac simulation using Finitewave. We will create a 2D mesh of
cardiac tissue, define the tissue properties, set up a model, apply
stimulation, and run the simulation.

.. contents:: Table of Contents
   :local:
   :depth: 3

Cardiac Tissue
----------------

The ``CardiacTissue`` class is used to represent myocardial tissue and its
structural properties in simulations. It includes several key attributes that
define the characteristics and behavior of the cardiac mesh used in
finite-difference calculations.

First, import the necessary libraries:

.. code-block:: Python

    import finitewave as fw
    import numpy as np
    import matplotlib.pyplot as plt


Initialize a 100x100 mesh with all nodes set to 1 (healthy cardiac tissue).
Add empty nodes (0) at the mesh edges to simulate boundaries.

.. code-block:: Python

    n = 100
    tissue = fw.CardiacTissue2D([n, n])

Mesh
""""

The ``mesh`` attribute is a mesh consisting of nodes, which
represent the myocardial medium. The distance between neighboring nodes is
defined by the spatial step (``dr``) parameter of the model. The nodes in the
mesh are used to represent different types of tissue and their properties:

* ``0``: Empty node, representing the absence of cardiac tissue.
* ``1``: Healthy cardiac tissue, which supports wave propagation.
* ``2``: Fibrotic or infarcted tissue, representing damaged or non-conductive areas.

Nodes marked as ``0`` and ``2`` are treated similarly as isolated nodes with no
flux through their boundaries. These different notations help distinguish
between areas of healthy tissue, empty spaces, and regions of fibrosis or
infarction.

.. note::

    To satisfy boundary conditions, every Finitewave mesh must include boundary 
    nodes (marked as ``0``). This can be easily achieved using the
    ``add_boundaries()`` method, which automatically adds rows of empty nodes
    around the edges of the mesh. You should apply this method if you modify the
    ``mesh``, for example by adding fibrosis.

You can also utilize ``0`` nodes to define complex geometries and pathways,
or to model organ-level structures. For example, to simulate the
electrophysiological activity of the heart, you can create a 3D array
where ``1`` represents cardiac tissue, and ``0`` represents everything outside
of that geometry.

Conductivity
""""""""""""

The conductivity attribute defines the local conductivity of the tissue and is
represented as an array of coefficients ranging from ``0.0`` to ``1.0`` for
each node in the mesh. It proportionally decreases the diffusion coefficient
locally, thereby slowing down the wave propagation in specific areas defined
by the user. This is useful for modeling heterogeneous tissue properties,
such as regions of impaired conduction due to ischemia or fibrosis.

.. code-block:: Python

    # Set conductivity to 0.5 in the middle of the mesh
    tissue.conductivity = np.ones([n, n])
    tissue.conductivity[n//4: 3 * n//4, n//4: 3 * n//4] = 0.5

Fibers
""""""

Another important attribute, ``fibers``, is used to define the anisotropic
properties of cardiac tissue. This attribute is represented as a 3D array
(for 2D tissue) or a 4D array (for 3D tissue), with each node containing a 2D
or 3D vector that specifies the fiber orientation at that specific position.
The anisotropic properties of cardiac tissue mean that the wave propagation
speed varies depending on the fiber orientation.

.. code-block:: Python

    # Fibers orientated along the x-axis
    tissue.fibers = np.zeros([n, n, 2])
    tissue.fibers[:, :, 0] = 1
    tissue.fibers[:, :, 1] = 0

Cardiac Models
----------------

Finitewave currently includes three built-in models for 2D and 3D simulations.
Each model represents the cardiac electrophysiological activity of a single
cell, which can be combined using parabolic equations to form complex 2D or 3D
cardiac tissue models.

.. code-block:: Python

    # Set up Aliev-Panfilov model to perform simulations
    aliev_panfilov = fw.AlievPanfilov2D()
    aliev_panfilov.dt = 0.01                # time step
    aliev_panfilov.dr = 0.25                # space step
    aliev_panfilov.t_max = 10               # simulation time

We use an explicit finite-difference scheme, which requires maintaining an
appropriate ``dt/dr`` ratio. For Aliev-Panfilov model, the recommended
calculation parameters for time and space steps are ``dt = 0.01`` and
``dr = 0.25``. You can increase ``dt`` to ``0.02`` to speed up calculations,
but always verify the stability of your numerical scheme, as instability will
lead to incorrect simulation results.

Available models
"""""""""""""""""""""""""""

Currently, finitewave includes three built-in models for 2D and 3D simulations,
but you can easily add your own models by extending the base class and
implementing the necessary methods.

+-----------------+---------------------------------------------------------------+
| Model           | Description                                                   | 
+=================+===============================================================+
| Aliev-Panfilov  | A phenomenological two-variable model for cardiac simulations |
+-----------------+---------------------------------------------------------------+
| Luo-Rudy 1991   | An ionic model for cardiac simulations                        |
+-----------------+---------------------------------------------------------------+ 
| TP06            | An ionic model for cardiac simulations                        |
+-----------------+---------------------------------------------------------------+


Stimulations
------------

To simulate the electrical activity of the heart, you need to apply a stimulus
to the tissue. This can be done by setting the voltage or current at specific
nodes in the mesh.

Voltage Stimulation
"""""""""""""""""""

``StimVoltage`` class allows directly sets voltage values at the nodes within
the stimulation area, triggering wave propagation from this region.

.. code-block:: Python

    stim_voltage = fw.StimVoltageCoord2D(time=0,
                                         volt_value=1,
                                         x1=1, x2=n-1, y1=1, y2=3)

Current Stimulation
"""""""""""""""""""

``StimCurrent`` class allows you to apply a current value and stimulation
duration to accumulate potential, leading to wave propagation. Current
stimulation offers more flexibility and is more physiologically accurate, as
it simulates the activity of external electrodes.

.. code-block:: Python

    stim_current = fw.StimCurrentCoord2D(time=0,
                                         curr_value=0.15,
                                         curr_time=1,
                                         x1=1, x2=n-1, y1=1, y2=3)

Stimulation Matrix
"""""""""""""""""""

By default, the stimulation area is defined as a rectangle
(``x1:x2, y1:y2, [z1:z2]``), but you can also define a custom Boolean array to
specify the nodes to be stimulated. This allows you to create complex
stimulation patterns.

.. code-block:: Python
    
    # Stimulate a 6x6 area in the middle of the mesh
    stim_matrix = np.zeros([n, n], dtype=bool)
    stim_matrix[n//2 - 3: n//2 + 3 , n//2 - 3: n//2 + 3] = True
    stim_current_matrix = fw.StimCurrentMatrix2D(time=0,
                                                 curr_value=0.15,
                                                 curr_time=1,
                                                 matrix=stim_matrix))

.. note::

    A very small stimulation area may lead to unsuccessful stimulation
    due to a source-sink mismatch.

Stimulation Sequence
"""""""""""""""""""""

The ``CardiacModel`` class uses the ``StimSequence`` class to manage the
stimulation sequence. This class allows you to add multiple stimulations to
the model, which can be useful for simulating complex stimulation protocols
(e.g., a high-pacing protocol).

.. code-block:: Python

    stim_sequence = fw.StimSequence()

    for i in range(0, 100, 10):
        stim_sequence.add_stim(fw.StimVoltageCoord2D(time=i,
                                                     volt_value=1,
                                                     x1=1, x2=n-1, y1=1, y2=3))

Trackers
--------

Trackers are used to record the state of the model during the simulation. They
can be used to monitor the wavefront propagation, visualize the activation
times, or analyze the wavefront dynamics. Full details on how to use trackers
can be found in the documentation and examples.

.. code-block:: Python

    # set up activation time tracker:
    act_time_tracker = fw.ActivationTime2DTracker()
    act_time_tracker.threshold = 0.5
    act_time_tracker.step = 100  # calculate activation time every 100 steps


Tracker Parameters
""""""""""""""""""

Trackers have several parameters that can be adjusted to customize their
behavior:

* ``start_time``: The time at which the tracker starts recording data.
* ``end_time``: The time at which the tracker stops recording data.
* ``step``: The number of steps between each data recording.

.. note:: 
    
    The ``step`` parameter is used to control the *frequency* of data
    recording (should be ``int``). But the ``start_time`` and ``end_time``
    parameters are used to specify the *time* interval during which the tracker
    will record data.

The ``output`` property of the tracker class returns the formatted data
recorded during the simulation. This data can be used for further analysis
or visualization.

Each tracker has its own set of parameters that can be adjusted to customize
its behavior. For example, the ``ActivationTime2DTracker`` class has a
``threshold`` parameter that defines the activation threshold for the nodes.

Multiple Trackers
"""""""""""""""""

The ``CardiacModel`` class uses the ``TrackerSequence`` class to manage the
trackers. This class allows you to add multiple trackers to the model to
monitor different aspects of the simulation. For example, you can track the
activation time for all nodes, and the action potential for a specific node.

.. code-block:: Python
    
    # set up first activation time tracker:
    act_time_tracker = fw.ActivationTime2DTracker()
    act_time_tracker.threshold = 0.5
    act_time_tracker.step = 100  # calculate activation time every 100 steps

    # set up action potential tracker for a specific node:
    action_pot_tracker = fw.ActionPotential2DTracker()
    action_pot_tracker.cell_ind = [30, 30]

    tracker_sequence = fw.TrackerSequence()
    tracker_sequence.add_tracker(act_time_tracker)
    tracker_sequence.add_tracker(action_pot_tracker)


Building pipeline
-----------------

Now that we have all the necessary components, we can build the simulation
pipeline by setting the tissue, model, stimulations, and trackers.

.. code-block:: Python

    aliev_panfilov.cardiac_tissue = tissue
    aliev_panfilov.stim_sequence = stim_sequence
    aliev_panfilov.tracker_sequence = tracker_sequence

Finitewave contains other functionalities that can be used to customize the
simulation pipeline, such as loading and saving model states or adding custom
commands to the simulation loop. For more information, refer to the full
documentation.


Run the simulation
""""""""""""""""""

Finally, we can run the simulation by calling the ``run()`` method of the
``AlievPanfilov2D`` model.

.. code-block:: Python

    aliev_panfilov.run()

    plt.imshow(aliev_panfilov.u, cmap='coolwarm')
    plt.show()


Simplified pipeline
-------------------

Here is a simplified version of the simulation pipeline that combines all the
steps described above:

.. code:: Python
    
    import numpy as np
    import matplotlib.pyplot as plt
    import finitewave as fw
    
    # set up the tissue:
    n = 100
    tissue = fw.CardiacTissue2D([n, n])
    # set up the stimulation:
    stim_sequence = fw.StimSequence()
    stim_sequence.add_stim(fw.StimVoltageCoord2D(time=0,
                                                volt_value=1,
                                                x1=1, x2=n-1, y1=1, y2=3))
    # set up the tracker:
    act_time_tracker = fw.ActivationTime2DTracker()
    act_time_tracker.threshold = 0.5
    act_time_tracker.step = 100

    tracker_sequence = fw.TrackerSequence()
    tracker_sequence.add_tracker(act_time_tracker)
    
    # set up the model
    aliev_panfilov = fw.AlievPanfilov2D()
    aliev_panfilov.dt = 0.01
    aliev_panfilov.dr = 0.25
    aliev_panfilov.t_max = 10
    
    # set up pipeline
    aliev_panfilov.cardiac_tissue = tissue
    aliev_panfilov.stim_sequence = stim_sequence
    aliev_panfilov.tracker_sequence = tracker_sequence
    
    # run model
    aliev_panfilov.run()
    
    # show output
    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(aliev_panfilov.u, cmap='coolwarm')
    axs[0].set_title("u")

    axs[1].imshow(act_time_tracker.output, cmap='viridis')
    axs[1].set_title("Activation time")

    fig.suptitle("Aliev-Panfilov 2D isotropic")
    plt.tight_layout()
    plt.show()

.. The output should look like this:

.. .. image-sg:: /usage/images/quick_start_001.png
..   :alt: Aliev-Panfilov 2D model
..   :srcset: /usage/images/quick_start_001.png
..   :class: sphx-glr-single-img

