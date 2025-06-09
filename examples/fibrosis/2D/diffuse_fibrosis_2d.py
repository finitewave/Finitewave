"""
2D Diffuse Fibrosis Pattern (20% Density)
=========================================

This example demonstrates how to generate a 2D cardiac tissue with 
a diffuse fibrosis pattern using the `Diffuse2DPattern` class in Finitewave.

Fibrotic tissue regions are marked as non-conductive areas in the mesh,
and this affects wave propagation in subsequent simulations.

Setup:
------
- Grid size: 200 × 200
- Fibrosis type: Diffuse (random spatial distribution)
- Fibrosis density: 20% (i.e., 20% of cells are fibrotic/non-conductive)

Visualization:
--------------
The generated tissue is shown as a 2D image:
- Green cells represent healthy (conductive) tissue
- Yellow cells represent fibrotic (non-conductive) areas

This mesh can be used in simulations to study how diffuse fibrosis alters
electrical propagation, reentry, and arrhythmogenesis.
"""

import matplotlib.pyplot as plt
import finitewave as fw

n = 200
# create mesh
tissue = fw.CardiacTissue2D((n, n))
tissue.add_pattern(fw.Diffuse2DPattern(0.2))

plt.title("2D Diffuse Fibrosis Medium with 20% Fibrosis Density")
plt.imshow(tissue.mesh)
plt.colorbar()
plt.show()