from pathlib import Path
import pyvista as pv


path = Path("/Users/arstanbekokenov/Projects/Finitewave/examples/data/atrial_mesh")
vtk_mesh = pv.read(path / "Mesh_31475951.vtk")

vtk_mesh.points /= 1000

plotter = pv.Plotter()
plotter.add_mesh(vtk_mesh)

camera_position = None

def track_camera(*args):
    global camera_position
    camera_position = plotter.camera_position
    print(camera_position)

plotter.iren.add_observer("EndInteractionEvent", track_camera)

plotter.show()
