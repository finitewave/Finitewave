from pathlib import Path
import numpy as np
import pyvista as pv
import natsort
from tqdm import tqdm

import finitewave as fw


path_data = Path("/Users/arstanbekokenov/Projects/Finitewave/Tutorials/frames")

path = Path("/Users/arstanbekokenov/Projects/Finitewave/examples/data/atrial_mesh")
# vtk_mesh = pv.read(path / "Mesh_31475951.vtk")
vtk_mesh = pv.read(Path("/Users/arstanbekokenov/Projects/Finitewave/Tutorials", "atrial_remesh.vtk"))

coords = vtk_mesh.points / 1000
elems = vtk_mesh.faces.reshape(-1, 4)[:, 1:4]

camera_position_1 = [(-0.07906315141255561, -0.15772625997191111, 0.12826262984316597),
                     (-0.013406300329890016, -0.02643289860487992, 0.05208395241158073),
                     (0.21447266621021388, 0.40772452095276956, 0.8875596827608367)]
camera_position_2 = [(-0.14023626476133416, -0.02074214846899797, -0.053905364095229676),
                     (-0.013406300329890016, -0.02643289860487992, 0.05208395241158073),
                     (-0.24394228286762693, 0.907988376486401, 0.34066005165695734)]

pv.global_theme.transparent_background = True

# files = natsort.natsorted(list(path_data.glob("*.npy")))
# print(f"Found {len(files)} files")

# grid = fw.PyVistaSurfaceGrid(coords, elems)
# grid["u"] = np.zeros(coords.shape[0])
# grid["u"] = np.load(files[0])

# pl = pv.Plotter(off_screen=True, window_size=(1920, 1920))
# # Open a movie file
# pl.open_movie("atrial_lat.gif", framerate=24)

# # Add initial mesh
# pl.add_mesh(grid, scalars='u', cmap="inferno")
# pl.show(auto_close=False)
# pl.camera_position = camera_position
# # Run through each frame
# pl.write_frame()  # write initial data

# # Update scalars on each frame
# for i in tqdm(range(len(files)), desc="Building animation"):
#     grid["u"] = np.load(files[i])
#     pl.write_frame()

# pl.close()

image_builder = fw.Image3DBuilder()
image_builder.grid = fw.PyVistaSurfaceGrid(coords, elems)
image_builder.grid["u"] = np.zeros(coords.shape[0])
image_builder.scalar_name = "u"
image_builder.collect_images(path_data)
image_builder.build_scene(clim=[-90, 10], cmap="inferno", window_size=(1000, 1000), show_scalar_bar=False)


import imageio

path = Path("atrial_lat").with_suffix(".gif")
total_frames = image_builder.total_frames
with imageio.get_writer(str(path), mode='I', fps=10, plugin="pyav", loop=0) as writer:
    for i in tqdm(range(total_frames), desc="Building animation"):
        if i < 50:
            camera_position = camera_position_1
        else:
            camera_position = camera_position_2
        img = image_builder.generate_image(i, camera_position=camera_position)
        writer.append_data(img)
