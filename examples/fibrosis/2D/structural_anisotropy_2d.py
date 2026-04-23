"""
Structural Anisotropy in 2D Due to Interstitial Fibrosis
=========================================================

This example demonstrates how interstitial fibrosis can create structural 
anisotropy in a 2D cardiac tissue. The fibrotic pattern causes directionally 
preferential conduction, leading to an elliptical spread of excitation from a 
point stimulus.

Unlike fiber-based anisotropy, which is driven by directional conductivity, 
this anisotropy arises purely from the geometry of the fibrotic microstructure.

Setup:
------
- Tissue: 2D grid of size 400×400
- Fibrosis:
    • Type: Interstitial (structured, linear obstacles)
    • Density: 15%
    • Strand size: 1 pixel wide × 4 pixels long (aligned in j-direction)
- Stimulus:
    • Type: Voltage stimulus
    • Location: Center of the tissue
    • Shape: Square (10×10 pixels)
    • Time: Applied at t = 0 ms

Model:
------
- Aliev-Panfilov 2D reaction-diffusion model
- Simulation time: 30 ms
- Time step: 0.01 ms
- Spatial resolution: 0.25 mm

Observation:
------------
Due to the aligned fibrotic obstacles, the excitation wavefront becomes 
elliptical, spreading more easily in the direction perpendicular
to the fibrosis strands. This mimics real-world structural anisotropy seen 
in interstitial fibrosis.

Applications:
-------------
This example is useful for exploring:
- Structural sources of conduction anisotropy
- Functional impact of interstitial fibrosis geometry
- Wavefront deformation and vulnerability to reentry

"""

import numpy as np
import matplotlib.pyplot as plt
import finitewave as fw

n = 400
# create mesh
tissue = fw.CardiacTissue2D((n, n))
tissue.add_pattern(fw.Structural2DPattern(density=0.15, length_i=1, length_j=4, x1=0, x2=n, y1=0, y2=n))

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, n//2 - 5, n//2 + 5,
                                             n//2 - 5, n//2 + 5))

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
plt.title("Structural Anisotropy 2D")
plt.imshow(aliev_panfilov.u)
plt.colorbar()
plt.show()
