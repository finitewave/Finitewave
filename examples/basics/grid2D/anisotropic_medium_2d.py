
"""
Aliev-Panfilov 2D Model (Anisotropic)
=====================================

Overview:
---------
This example demonstrates how to simulate the Aliev-Panfilov model in a 
two-dimensional anisotropic cardiac tissue. Unlike the isotropic case, 
anisotropy is introduced by specifying a fiber orientation array, which 
modifies the diffusion properties of the tissue.

Simulation Setup:
-----------------
- Tissue Grid: A 400×400 cardiac tissue domain is created.
- Anisotropic Diffusion: Fiber orientation is set using a direction field.
- Fiber Orientation: Defined by an angle alpha = 0.25 * pi.
- Stimulation: A localized stimulus is applied at the center of the domain.
- Time and Space Resolution:
  - Temporal step (dt): 0.01 
  - Spatial resolution (dr): 0.25 
  - Total simulation time (t_max): 30 

Execution:
----------
1. Create a 2D cardiac tissue grid with fiber orientation.
2. Define and apply a stimulus at the center.
3. Set up and initialize the Aliev-Panfilov model.
4. Run the simulation to compute wave propagation in an anisotropic medium.
5. Visualize the membrane potential distribution at the final timestep.

Anisotropic Diffusion:
----------------------
Anisotropy is implemented by defining a fiber orientation field for the 
CardiacTissue object. The model automatically selects the appropriate stencil 
to calculate the diffusion term based on fiber direction.

Visualization:
--------------
The final membrane potential distribution is displayed using matplotlib, 
showing how the excitation wave propagates in the anisotropic medium.
"""


import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 256
# fiber orientation angle
tissue = fw.CardiacTissueGrid([n, n])
alpha = np.pi / 4
tissue.mesh += (np.random.random(tissue.mesh.shape) < 0.2)
# add fibers orientation vectors
tissue.fibers = np.zeros([n, n, 2])
tissue.fibers[:, :, 0] = np.cos(alpha)
tissue.fibers[:, :, 1] = np.sin(alpha)

diffusion_model = fw.GridAssembler()
diffusion_model.stencil = fw.SymmetricStencil()

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(time=0, volt_value=1,
                                           x_min=n//2 - 5, x_max=n//2 + 5,
                                           y_min=n//2 - 5, y_max=n//2 + 5))

simulation = fw.CardiacSimulation()
simulation.dt = 0.01
simulation.dr = 0.1
simulation.t_max = 30
# add the tissue and the stim parameters to the model object:
simulation.cardiac_tissue = tissue
simulation.cardiac_model = fw.Courtemanche()
simulation.cardiac_model.step = 1
simulation.stim_sequence = stim_sequence
simulation.diffusion_model = diffusion_model

# run the model:
simulation.run()

# visualize the results:
plt.imshow(simulation.cardiac_model.u, cmap='jet', origin='lower')
plt.colorbar(label='Transmembrane Potential (u)')
plt.title('Aliev-Panfilov Model - Transmembrane Potential')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
