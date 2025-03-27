"""
Running the Aliev-Panfilov Model in 2D
======================================

Overview:
---------
This example demonstrates how to run a basic 2D simulation of the 
Aliev-Panfilov model using the Finitewave framework. It simulates 
electrical excitation in cardiac tissue following a localized stimulus.

Simulation Setup:
-----------------
- Tissue Grid: A 300×300 cardiac tissue domain.
- Stimulation:
  - A square side stimulus is applied at t = 0.
- Time and Space Resolution:
  - Temporal step (dt): 0.01
  - Spatial resolution (dr): 0.25
  - Total simulation time (t_max): 20

Execution:
----------
1. Create a 2D cardiac tissue grid.
2. Apply a stimulus along the upper boundary to initiate excitation.
3. Set up and run the Aliev-Panfilov model.
4. Visualize the final transmembrane potential map.

Application:
------------
- This example serves as a minimal working simulation for testing and demonstration.
- It can be used as a template for:
  - Adding trackers
  - Extending to anisotropic or heterogeneous tissues
  - Exploring basic wave propagation

Output:
-------
Displays the membrane potential distribution (`u`) at the end of the simulation using matplotlib.

"""

import matplotlib.pyplot as plt

import finitewave as fw

# create a tissue of size 300x300 with cardiomycytes:
n = 300
tissue = fw.CardiacTissue2D([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, n, 0, 5))

# create model object and set up parameters:
aliev_panfilov = fw.AlievPanfilov2D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 20
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
