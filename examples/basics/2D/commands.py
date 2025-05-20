"""
Fenton-Karma 2D Model (Interrupt via Custom Command)
====================================================

Overview:
---------
This example demonstrates how to use the `Command` and `CommandSequence` interfaces in Finitewave
to inject custom logic into a cardiac electrophysiology simulation. Specifically, we interrupt the
simulation when the wave of excitation reaches the far edge of the tissue. This demonstrates how
to trigger actions based on the state of the system.

Simulation Setup:
-----------------
- Tissue Grid: A 300×300 cardiac tissue domain.
- Mesh: Entire domain is active tissue (`1.0` values).
- Model: Fenton-Karma 2D model is used for wave propagation.
- Stimulation: A voltage stimulus is applied along the entire left edge.
- Time and Space Resolution:
  - Temporal step (dt): 0.01 ms
  - Spatial resolution (dr): 0.25 mm
  - Total simulation time (t_max): 1000 ms

Command Usage:
--------------
- Custom command `InterruptCommand` inherits from `Command`.
- At every 10 ms of simulation time (from 0 to 190 ms), the command checks if the wave has 
  reached the far-right side (`x = 298`).
- If any value of the membrane potential exceeds 0.5 on this edge, the simulation is terminated
  early by setting `model.step = np.inf`.

Execution:
----------
1. Initialize a square 2D tissue with a full mesh of excitable tissue.
2. Apply a uniform voltage stimulus along the leftmost edge (columns `0–1`).
3. Set up a sequence of `InterruptCommand` checks at regular intervals.
4. Run the simulation. It will self-interrupt once the wave reaches the far side.

Effect of Custom Command:
-------------------------
This feature is useful for:
- Saving computational time by stopping early based on user-defined conditions.
- Triggering intermediate analysis, adaptive pacing, or feedback-based protocols.
- Debugging or validation of wave speed and tissue responsiveness.

Visualization:
--------------
No visualization is included in this example, but users can integrate `matplotlib` or export
model states using built-in Finitewave I/O utilities.

Notes:
------
- The `Command` and `CommandSequence` classes allow flexible integration of logic and control flow
  without modifying the core model.
- This technique is extendable to more complex use cases such as region-specific feedback, pacing adjustment, 
  or custom logging.

"""

import numpy as np

import finitewave as fw


# number of nodes on the side
n = 300

tissue = fw.CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n], dtype=float)
# add empty nodes on the sides (elems = 0):

# create model object:
fenton_karma = fw.FentonKarma2D()
# set up numerical parameters:
fenton_karma.dt    = 0.01
fenton_karma.dr    = 0.25
fenton_karma.t_max = 1000

# Define the command:
class InterruptCommand(fw.Command):
    def execute(self, model):
        if np.any(model.u[:, 298] > 0.5):
             # increase the calculation step to stop the execution loop.
             model.step = np.inf
             print ("Propagation wave reached the opposite side. Stop calculation.")

# We want to check the opposite side every 10 time units.
# Thus we have a list of commands with the same method but different times.
command_sequence = fw.CommandSequence()
for i in range(0, 200, 10):
    command_sequence.add_command(InterruptCommand(i))

fenton_karma.command_sequence = command_sequence

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, n, 0, 5))

# add the tissue and the stim parameters to the model object:
fenton_karma.cardiac_tissue = tissue
fenton_karma.stim_sequence  = stim_sequence

fenton_karma.run()