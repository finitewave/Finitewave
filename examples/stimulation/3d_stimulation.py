"""
Stimulation in 3D
==================================

Overview:
---------
This example demonstrates how to apply two opposite planar waves in 3D tissue using:
- `StimVoltageCoord3D`: voltage stimulation with spatial bounds (`x1`, `x2`, `y1`, `y2`, `z1`, `z2`)
- `StimCurrentMatrix3D`: matrix-based current stimulation with a 3D boolean array.

The example highlights that 3D stimulation setup is identical to 2D, 
with the only difference being the inclusion of the Z-axis (`z1`, `z2` or 3D matrix).

Simulation Setup:
-----------------
- Tissue: 3D slab (200×200×10)
- Stimulus 1: Voltage-based planar wave on the left face at `t=0`
- Stimulus 2: Current-based planar wave on the right face at `t=0`
- Duration: 10 time units

Application:
------------
- Demonstrates stimulation syntax for 3D using both coordinate and matrix methods.
- Visualizes resulting voltage (`u`) distribution in 3D using PyVista.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import finitewave as fw

# tissue setup
nx = 200
ny = 200
nz = 30
tissue = fw.CardiacTissue3D([nx, ny, nz])
tissue.mesh = np.ones((nx, ny, nz), dtype=np.uint8)

# stimulus 1: VoltageCoord3D on left face
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(
    fw.StimVoltageCoord3D(time=0, volt_value=1.0,
                          x1=0, x2=5,
                          y1=0, y2=ny,
                          z1=0, z2=nz)
)

# stimulus 2: CurrentMatrix3D on right face
stim_matrix = np.zeros((nx, ny, nz), dtype=bool)
stim_matrix[nx - 5:nx, :, :] = True  # Right face
stim_sequence.add_stim(
    fw.StimCurrentMatrix3D(time=0, curr_value=10, duration=0.5, matrix=stim_matrix)
)

# model setup:
model = fw.AlievPanfilov3D()
model.dt = 0.01
model.dr = 0.25
model.t_max = 10
model.cardiac_tissue = tissue
model.stim_sequence = stim_sequence

# run the model:
model.run()

# visualization with PyVista:
mesh_builder = fw.VisMeshBuilder3D()
mesh_grid = mesh_builder.build_mesh(tissue.mesh)
mesh_grid = mesh_builder.add_scalar(model.u, name='Membrane Potential (u)')
mesh_grid.plot(cmap='viridis', clim=[0, 1])