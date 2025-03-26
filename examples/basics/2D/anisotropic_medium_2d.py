
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
n = 400
# fiber orientation angle
alpha = 0.25 * np.pi
tissue = fw.CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n])
tissue.add_boundaries()
# add fibers orientation vectors
tissue.fibers = np.zeros([n, n, 2])
tissue.fibers[:, :, 0] = np.cos(alpha)
tissue.fibers[:, :, 1] = np.sin(alpha)

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, n//2 - 3, n//2 + 3,
                                                n//2 - 3, n//2 + 3))

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 30
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

aliev_panfilov.run()

# show the potential map at the end of calculations:
plt.figure()
plt.imshow(aliev_panfilov.u)
plt.colorbar()
plt.show()
