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
import mlx.core as mx


# create a tissue of size 400x400 with cardiomycytes:
n = 400
tissue = fw.CardiacTissueGrid([n, n], dr=0.25)

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(time=0, volt_value=1,
                                           x_min=n//2 - 3, x_max=n//2 + 3,
                                           y_min=n//2 - 3, y_max=n//2 + 3))

# create model object and set up parameters:
simulation = fw.CardiacSimulation()
simulation.dt = 0.01
simulation.t_max = 0.02
simulation.cardiac_model = fw.AlievPanfilovMLX(memory_save=True)
simulation.cardiac_tissue = tissue
# simulation.stim_sequence = stim_sequence
simulation.initialize()

u = np.zeros((n, n), dtype=np.float32)
u[n//2 - 3:n//2 + 3, n//2 - 3:n//2 + 3] = 1.0

simulation.cardiac_model.u = mx.array(u.flatten(), dtype=mx.float32)

# run the model:
simulation.run(initialize=False)

print("Simulation completed.")
u = np.array(simulation.cardiac_model.u).reshape(tissue.mesh.shape)
v = np.array(simulation.cardiac_model.v).reshape(tissue.mesh.shape)

# show the potential map at the end of calculations:
fig, axs = plt.subplots(ncols=2)
im = axs[0].imshow(u, cmap="inferno")
axs[0].set_title("Membrane Potential (u)")
im = axs[1].imshow(v, cmap="inferno")
axs[1].set_title("Recovery Variable (v)")
plt.show()