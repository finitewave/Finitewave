from pathlib import Path
import numpy as np
import pyvista as pv
import natsort
from tqdm import tqdm

import finitewave as fw


path_data = Path("/Users/arstanbekokenov/Projects/Finitewave/lat_data")

path = Path("/Users/arstanbekokenov/Projects/Finitewave/examples/data/atrial_mesh")
vtk_mesh = pv.read(path / "Mesh_10954794.vtk")

coords = vtk_mesh.points / 1000
elems = vtk_mesh.faces.reshape(-1, 4)[:, 1:4]

camera_position = [
    (55.25059642294556, 207.60117292709876, -65.85503184893892),
    (-32.90564727783203, -27.661887526512146, 75.8362808227539),
    (-0.18539692142201003, 0.5570278116024722, 0.8095356685338835)
]

pv.global_theme.transparent_background = True

files = natsort.natsorted(list(path_data.glob("*.npy")))
print(f"Found {len(files)} files")

grid = fw.PyVistaSurfaceGrid(coords, elems)
grid["u"] = np.zeros(coords.shape[0])

pl = pv.Plotter(off_screen=True, window_size=(1920, 1920))
# Open a movie file
pl.open_movie("atrial_lat.gif", framerate=24)

# Add initial mesh
pl.add_mesh(grid, scalars='u', clim=[0, 28], cmap="inferno")
pl.show(auto_close=False)
pl.camera_position = camera_position
# Run through each frame
pl.write_frame()  # write initial data

# Update scalars on each frame
for i in tqdm(range(len(files)), desc="Building animation"):
    grid["u"] = np.load(files[i])
    pl.write_frame()

pl.close()
