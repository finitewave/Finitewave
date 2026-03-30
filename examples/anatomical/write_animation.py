from pathlib import Path
import numpy as np
import pyvista as pv

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

anim_builder = fw.Animation3DBuilder()
anim_builder.path = path_data
anim_builder.prog_bar = True
anim_builder.write(coords=coords, elems=elems, elem_type='Tri', clim=None,
                   format="gif", camera_position=camera_position,
                   cmap="magma", window_size=(1920, 1920),
                   step=4)
