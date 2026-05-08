"""
Aliev-Panfilov 2D Model (Isotropic)
====================================

Overview:
---------
This example demonstrates how to simulate the Aliev-Panfilov model in a 
two-dimensional isotropic medium using the Finitewave framework. The model 
describes the propagation of electrical waves in excitable media, such as 
cardiac tissue, and captures fundamental excitation and recovery dynamics.

Simulation Setup:
-----------------
- Tissue Grid: A 400×400 homogeneous cardiac tissue is created.
- Isotropic Stencil: Diffusion is uniform in all directions.
- Stimulation: A localized stimulus is applied at the center of the domain.
- Time and Space Resolution:
  - Temporal step (dt): 0.01
  - Spatial resolution (dr): 0.25
  - Total simulation time (t_max): 30

Execution:
----------
1. Create a 2D cardiac tissue grid.
2. Define and apply a stimulus at the center.
3. Set up and initialize the Aliev-Panfilov model.
4. Run the simulation to compute wave propagation.
5. Visualize the membrane potential map at the final timestep.

Visualization:
--------------
The final membrane potential distribution is displayed using `matplotlib`, 
showing the resulting excitation wave pattern.
"""

import matplotlib.pyplot as plt

import finitewave as fw
import numpy as np

stim_prepacing = fw.StimSingleCell(dt=0.005)
stim_prepacing.add_stim(n_beats=30, cycle_length=1000., curr_value=20., duration=2.)
stim_prepacing.add_stim(n_beats=30, cycle_length=500., curr_value=20., duration=2.)

# create model object and set up parameters
courtemanche = fw.Courtemanche()
courtemanche.prepacing(stim_prepacing)


# create a tissue of size 400x400 with cardiomycytes:
n = 400
tissue = fw.CardiacTissueGrid([n, n], dr=0.25)

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(time=0, volt_value=1,
                                           x_min=n//2 - 3, x_max=n//2 + 3,
                                           y_min=n//2 - 3, y_max=n//2 + 3))

# create model object and set up parameters:
simulation = fw.CardiacSimulation(dt=0.01, t_max=50, backend="jax")
simulation.cardiac_model = courtemanche
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence

# run the model:
simulation.run()

u = simulation.cardiac_model.output("u")

# show the potential map at the end of calculations:
plt.figure()
plt.imshow(u, cmap='magma', origin="lower")
plt.colorbar(label="Membrane Potential")
plt.show()
