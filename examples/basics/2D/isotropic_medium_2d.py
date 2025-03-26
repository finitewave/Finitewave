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

# create a tissue of size 400x400 with cardiomycytes:
n = 400
tissue = fw.CardiacTissue2D([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, n//2 - 3, n//2 + 3,
                                             n//2 - 3, n//2 + 3))

# create model object and set up parameters:
aliev_panfilov = fw.AlievPanfilov2D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 30
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

# run the model:
aliev_panfilov.run()

# show the potential map at the end of calculations:
plt.figure()
plt.imshow(aliev_panfilov.u)
plt.colorbar()
plt.show()
