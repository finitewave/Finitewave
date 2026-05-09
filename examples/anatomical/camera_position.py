from pathlib import Path
import pyvista as pv
import numpy as np


# path = Path("/Users/arstanbekokenov/Projects/Finitewave/examples/data/atrial_mesh")
# vtk_mesh = pv.read(path / "Mesh_31475951.vtk")

vtk_mesh = pv.read(Path("/Users/arstanbekokenov/Projects/Finitewave/Tutorials", "atrial_remesh.vtk"))

vtk_mesh.points /= 1000

plotter = pv.Plotter()
plotter.add_mesh(vtk_mesh)

camera_position = None

def track_camera(*args):
    global camera_position
    camera_position = plotter.camera_position
    # print with 2 decimal places
    print(camera_position)


def callback(point):
    # Get closest point ID
    point_id = vtk_mesh.find_closest_point(point)
    print("Picked point ID:", point_id)

plotter.iren.add_observer("EndInteractionEvent", track_camera)
plotter.enable_point_picking(callback=callback, show_point=True)
plotter.show()
