"""
Left Ventricle Simulation with Anatomical Mesh and Fibers
----------------------------------------------------------

This example demonstrates how to simulate electrical activity in a
realistic left ventricular (LV) geometry using the Aliev-Panfilov
model in 3D.

The LV mesh and corresponding fiber orientations are loaded from
external data (available at https://zenodo.org/records/3890034).
The mesh is embedded in a regular grid, and fiber directions are
assigned to the myocardium using a vector field.

Stimulation is applied at the base of the ventricle to initiate 
activation, and wave propagation is visualized in 3D.

Data Requirements:
------------------
This example assumes the following files exist in the `data/` directory:
- `mesh.npy`: 3D binary array (1 = myocardium, 0 = empty)
- `fibers.npy`: Flattened array of fiber vectors (same shape as mesh[mesh > 0])

Simulation Setup:
-----------------
- Model: Aliev-Panfilov 3D
- Mesh: Realistic LV shape, embedded in a cubic grid
- Fibers: Anatomically derived vectors per voxel
- Stimulus:
    - Type: Voltage
    - Location: Basal region (first 20 z-slices)
    - Time: t = 0 
- Time step (dt): 0.01 
- Space step (dr): 0.25 
- Total time: 40 

Visualization:
--------------
- The scalar voltage field (`u`) is rendered in 3D using 
  Finitewave’s `VisMeshBuilder3D`.

Applications:
-------------
- Realistic whole-ventricle simulations
- Exploration of fiber-driven anisotropic conduction
- Foundation for further patient-specific modeling or ECG computation
"""


from pathlib import Path
import numpy as np

import finitewave as fw


path = Path(__file__).parent

# Load mesh as cubic array
mesh = np.load(path.joinpath("data", "mesh.npy"))

# Load fibers as cubic array
fibers_list = np.load(path.joinpath("data", "fibers.npy"))
fibers = np.zeros(mesh.shape + (3,), dtype=float)
fibers[mesh > 0] = fibers_list

# set up the tissue with fibers orientation:
tissue = fw.CardiacTissue3D(mesh.shape)
tissue.mesh = mesh
tissue.add_boundaries()
tissue.fibers = fibers

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, mesh.shape[0],
                                             0, mesh.shape[0],
                                             0, 20))

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 40
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
# initialize model: compute weights, add stimuls, trackers etc.
aliev_panfilov.run()

# visualize the ventricle in 3D
mesh_builder = fw.VisMeshBuilder3D()
mesh_grid = mesh_builder.build_mesh(tissue.mesh)
mesh_grid = mesh_builder.add_scalar(aliev_panfilov.u, 'u')
mesh_grid.plot(clim=[0, 1], cmap='viridis')