from pathlib import Path

from finitewave.tools.vis_mesh_builder_3d import VisMeshBuilder3D
from finitewave.core.tracker.tracker import Tracker


class VTKFrame3DTracker(Tracker):
    """
    A class for tracking and saving VTK frames in a 3D model.

    Attributes
    ----------
        file_name (str): The name of the saved frames.
        dir_name (str): The name of the folder where the frames will be saved.
        variable_name (str): The name of the target array to be tracked.
        file_type (str): The file type of the saved frames (".vtk" or ".vtu")
    """
    def __init__(self):
        super().__init__()
        self.file_name = "frame"
        self.dir_name = "vtk_frames"
        self.variable_name = ""
        self.file_type = ".vtk"

        self._frame_counter = 0

    def initialize(self, model):
        """
        Initializes the tracker with the model.

        Parameters
        -----------
            model (CardiacModel): The model to track.
        """
        self.model = model
        self._frame_counter = 0

        self.path = Path(self.path)

        if not self.path.joinpath(self.dir_name).exists():
            self.path.joinpath(self.dir_name).mkdir(parents=True)

        if self.variable_name == "":
            raise ValueError("Please specify the target array to be tracked.")

        if self.variable_name not in self.model.__dict__:
            raise ValueError(f"Array {self.variable_name} not found in model.")

    def _track(self):
        frame_name = self.path.joinpath(
            self.dir_name, f"{self.file_name}{self._frame_counter}"
            ).with_suffix(self.file_type)

        self.write_frame(frame_name)
        self._frame_counter += 1

    def write_frame(self, frame_name):
        """
        Writes a VTK frame to a file.

        Parameters
        -----------
            frame_name (str): The name of the frame file.
        """
        state_var = self.model.__dict__[self.variable_name]

        vtk_mesh_builder = VisMeshBuilder3D()
        vtk_mesh = vtk_mesh_builder.build_mesh(self.model.cardiac_tissue.mesh)
        vtk_mesh = vtk_mesh_builder.add_scalar(state_var, self.variable_name)
        vtk_mesh.save(frame_name)

    def write(self):
        """
        For compatibility with the Tracker class.
        """
        return super().write()
