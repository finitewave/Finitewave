
"""
Aliev-Panfilov 2D Model (Conductivity)
======================================

Overview:
---------
This example demonstrates how to simulate the Aliev-Panfilov model in a 
two-dimensional isotropic cardiac tissue with spatially varying conductivity. 
Conductivity variations affect wave propagation, simulating regions of different 
electrophysiological properties.

Simulation Setup:
-----------------
- Tissue Grid: A 400×400 cardiac tissue domain.
- Isotropic Diffusion: Conductivity is uniform within regions but varies across the tissue.
- Conductivity Variation: 
  - The default conductivity is set to 1.0.
  - The bottom-right quadrant (n/2:, n/2:) has reduced conductivity (0.3).
- Stimulation: A localized stimulus is applied at the center.
- Time and Space Resolution:
  - Temporal step (dt): 0.01 
  - Spatial resolution (dr): 0.25 
  - Total simulation time (t_max): 30 

Execution:
----------
1. Create a 2D cardiac tissue grid and define spatial conductivity variations.
2. Apply a stimulus at the center.
3. Set up and initialize the Aliev-Panfilov model.
4. Run the simulation to observe how conductivity affects wave propagation.
5. Visualize the final membrane potential distribution.

Effect of Conductivity:
-----------------------
The lower conductivity region slows down wave propagation, potentially leading 
to conduction block or reentrant wave formation. This feature is useful for modeling 
heterogeneous tissue properties such as fibrosis or ischemic regions.

Visualization:
--------------
The final membrane potential distribution is displayed using matplotlib, 
illustrating the impact of conductivity variations on wave propagation.
"""

import numpy as np
import matplotlib.pyplot as plt

import finitewave as fw

# create a tissue of size 400x400 with cardiomycytes:
n = 400
tissue = fw.CardiacTissue2D([n, n])
tissue.conductivity = np.ones([n, n], dtype=float)
tissue.conductivity[n//2:, n//2:] = 0.3

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1,
                                             n//2 - 3, n//2 + 3,
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
plt.imshow(aliev_panfilov.u)
plt.colorbar()
plt.show()
