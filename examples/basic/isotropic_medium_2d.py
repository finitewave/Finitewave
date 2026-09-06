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

# stim_prepacing = fw.StimSingleCell(dt=0.005)
# stim_prepacing.add_stim(n_beats=30, cycle_length=1000., curr_value=20., duration=2.)
# stim_prepacing.add_stim(n_beats=30, cycle_length=500., curr_value=20., duration=2.)

# create model object and set up parameters
cardiac_model = fw.Courtemanche()
# cardiac_model.prepacing(stim_prepacing)


# create a tissue of size 400x400 with cardiomycytes:
n = 400

# path = "/Users/arstanbekokenov/Projects/Fibrowave/simulations/data/PID09/segment/seg_mesh.npy"
# mesh = np.load(path)

# n, m, k = mesh.shape

# print("Mesh shape:", mesh.shape)

tissue = fw.CardiacTissueGrid((n, n), dr=0.25)
# tissue.mesh = mesh
# tissue.mesh += (np.random.rand(*tissue.mesh.shape) < 0.1).astype(int)  # add some fibrosis

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(time=0, volt_value=1,
                                           x_min=n//2 - 5, x_max=n//2 + 5,
                                           y_min=n//2 - 5, y_max=n//2 + 5))

# stim_sequence.add_stim(fw.StimVoltageCoord(time=0, volt_value=1,
#                                            x_min=0, x_max=n//5,
#                                            y_min=0, y_max=n//5,
#                                            z_min=0, z_max=n//5))

# create model object and set up parameters:
simulation = fw.CardiacSimulation(dt=0.01, t_max=50, backend="jax")
simulation.cardiac_model = cardiac_model
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence

# run the model:
simulation.run()

# plt.figure()
# plt.imshow(simulation.spatial_discretization.weights[0].toarray(), cmap='viridis', origin="lower")
# plt.colorbar(label="Weight")
# plt.show()

u = simulation.cardiac_model.output("u")

# show the potential map at the end of calculations:
plt.figure()
plt.imshow(u, cmap='magma', origin="lower")
plt.colorbar(label="Membrane Potential")
plt.show()
