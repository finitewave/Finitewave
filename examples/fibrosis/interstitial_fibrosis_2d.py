"""
2D Interstitial Fibrosis Pattern (20% Density, 4-Pixel Length)
==============================================================

This example demonstrates how to generate a 2D cardiac tissue with 
an interstitial fibrosis pattern using the `Structural2DPattern` class
from Finitewave.

Interstitial fibrosis is modeled as thin, linear fibrotic structures
or strands, typically aligned along fibers or tissue direction. These
structures act as barriers to conduction, affecting wave propagation.

Setup:
------
- Grid size: 200 × 200
- Fibrosis type: Interstitial (structured linear insertions)
- Fibrosis density: 20%
- Strand dimensions:
    • i-direction thickness: 1 pixel
    • j-direction length: 4 pixels
- Fibrosis applied uniformly over the whole domain

Visualization:
--------------
The generated tissue is shown as a 2D image:
- Green regions: healthy, conductive tissue
- Yellow linear elements: fibrotic, non-conductive strands (interstitial fibrosis)

Application:
------------
This type of structured pattern is useful for simulating how thin fibrotic
barriers affect action potential propagation, slow conduction, and create
substrates for reentrant activity.

"""


import matplotlib.pyplot as plt
import finitewave as fw

n = 200
# create mesh
tissue = fw.CardiacTissue2D((n, n))
tissue.add_pattern(fw.Structural2DPattern(density=0.2, length_i=1, length_j=4, x1=0, x2=n, y1=0, y2=n))

plt.title("2D Interstitial Fibrosis Medium with 20% Fibrosis Density and 4 pixels length")
plt.imshow(tissue.mesh)
plt.colorbar()
plt.show()