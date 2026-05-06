"""
Change initial conditions in the Luo-Rudy model 
======================================

Overview:
---------
This example demonstrates how to set spatially varying initial conditions 
in the Luo-Rudy 1991 cardiac model. Here we create a non-conductive obstacle in the center of the tissue 
and initialize an anatomical reentry by keeping the right half of the tissue refractory.

Simulation Setup:
-----------------
- Tissue Grid: A 256×256 cardiac tissue domain.
- Non-conductive Obstacle: A circular non-conductive region is created in the center of 
    the tissue to simulate an anatomical obstacle.
- Stimulation: A localized stimulus is applied to the left half of the upper border.
- Initial Conditions: The right half of the tissue is initialized in a refractory state by setting the gating 
    variables 'h' and 'j' to 0, while the left half is initialized to 0.9 (non-refractory).
- Time and Space Resolution:
    - Temporal step (dt): 0.01 
    - Spatial resolution (dr): 0.4 
    - Total simulation time (t_max): 200

Execution:
----------
1. Create a 2D cardiac tissue grid and define a non-conductive obstacle.
2. Apply a stimulus to the left half of the upper border.
3. Set up and initialize the Luo-Rudy 1991 model with spatially varying initial conditions.
4. Run the simulation to observe wave propagation and interaction with the obstacle.
5. Visualize the final membrane potential distribution, highlighting the non-conductive region.

How to use initial conditions:
------------------------------
Initial conditions can be set by defining attributes that start with 'init_' 
followed by the name of the model variable. In this example, we set 'init_h' and 'init_j' 
to create a refractory region in the right half of the tissue. This allows us to control 
the initial state of the tissue and observe how it affects wave propagation.
"""


import numpy as np
import matplotlib.pyplot as plt
import finitewave as fw

import numpy as np
import matplotlib.pyplot as plt
import finitewave as fw


n = 300
tissue = fw.CardiacTissue([n, n])

# Create a circular non-conductive obstacle in the center
x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

cx, cy = n // 2, n // 2
radius = 70

obstacle = (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2

tissue.mesh[obstacle] = 0 # non-conductive


stim_sequence = fw.StimSequence()

# Stimulate only the left half of the upper border
stim_sequence.add_stim(
    fw.StimVoltageCoord(
        time=0,
        volt_value=20,
        x1=0, x2=n // 2,
        y1=0, y2=n // 2,
    )
)

model = fw.LuoRudy91()
model.dt = 0.01
model.dr = 0.5
model.t_max = 200

model.cardiac_tissue = tissue
model.stim_sequence = stim_sequence

# Here we use initial conditions to keep the right half of the tissue refractory, 
# which will prevent wave propagation in that region until it is released.
#
# Variables starts with init_ are used as initial conditions for the corresponding model variables.
# You can set them to scalar values (all elements will have the same initial value) or to spatially varying arrays. 
# In this case, we set 'h' and 'j' to arrays that are 0 in the left half and 1 in the right half, 
# which corresponds to the refractory state in the Luo-Rudy model.
model.init_h = np.full_like(tissue.mesh, 0.9, dtype=np.float64)
model.init_h[n // 2:, :] = 0.0
model.init_j = np.full_like(tissue.mesh, 0.9, dtype=np.float64)
model.init_j[n // 2:, :] = 0.0

model.run()

# Show the potential map at the end of calculations.
# We also overlay the obstacle in gray to visualize its location relative to the wave propagation.
fig, axs = plt.subplots(ncols=1)
plt.rcParams["axes.titley"] = -0.25
obstacle_overlay = np.ma.masked_where(~obstacle, obstacle)
axs.imshow(model.u)
axs.imshow(obstacle_overlay, cmap='gray', alpha=1.0, vmin=0, vmax=1)
plt.tight_layout()
plt.show()