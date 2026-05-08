from pathlib import Path
import numpy as np

from finitewave.core import tissue

from .frame_tracker import FrameTracker
from finitewave.tools.animation_builder import (
    AnimationBuilder,
    Image2DBuilder,
    Image3DBuilder
)
from finitewave.tools.pyvista_grids import (
    PyVistaMeshGrid,
    PyVistaSurfaceGrid,
    PyVistaTetraGrid
)

from finitewave.core.numerics.fem.elements.element_type import ElementType


class AnimationTracker(FrameTracker):
    """
    A class to track and save frames of a 2D cardiac tissue simulation simulation
    for animation purposes.

    This tracker periodically saves the state of a specified target array from
    the simulation to disk as NumPy files, which can later be used to create
    animations.

    Attributes
    ----------
    animation_name : str
        The name of the directory where the animation frames will be saved.
    **kwargs : dict
        Additional keyword arguments for the FrameTracker base class.
    """
    def __init__(self, animation_name="animation", **kwargs):
        super().__init__(**kwargs)
        self.dir_name = animation_name

    @property
    def animation_name(self):
        return self.dir_name

    @animation_name.setter
    def animation_name(self, animation_name):
        self.dir_name = animation_name

    def write(self, prog_bar=True, fps=12, clim=[0, 1], cmap="RdBu_r", format="mp4",
              clear=True, **kwargs):
        """
        Creates an animation from the saved frames using the Animation2DBuilder
        class. Fibrosis and boundaries will be shown in black.

        Parameters
        ----------
        prog_bar : bool, optional
            Whether to show a progress bar during animation creation.
        fps : int, optional
            Frames per second for the output animation.
        clim : list, optional
            Color limits for the animation frames.
        cmap : str, optional
            Colormap to use for the animation.
        format : str, optional
            Format of the output animation ("mp4", "gif").
        clear : bool, optional
            Whether to clear the saved frames after creating the animation.
        **kwargs : dict
            Additional keyword arguments for the animation builder.
        """
        if self.aggregate:
            images = self.output
        else:
            images = None

        tissue = self.simulation.cardiac_tissue

        if tissue.mesh.ndim == 2:
            upscale_factor = kwargs.get("upscale_factor", 1)
            restore_input = not self.keep_shape
            image_builder = Image2DBuilder()
            image_builder.build_grid(tissue.mesh, restore_input, upscale_factor)
            
        else:
            window_size = kwargs.get("window_size", (1920, 1080))
            grid = self.build_3d_grid(tissue)
            grid[self.var_name] = np.zeros(tissue.mesh.shape, dtype=float)
            image_builder = Image3DBuilder()
            image_builder.scalar_name = self.var_name
            image_builder.grid = grid
            image_builder.build_scene(clim=clim, cmap=cmap, window_size=window_size)

        image_builder.collect_images(path=Path(self.path, self.dir_name), images=images)
        
        animation_builder = AnimationBuilder()
        animation_builder.image_builder = image_builder

        if format == "gif":
            writer = animation_builder.write_gif
        else:
            writer = animation_builder.write

        writer(path=self.path, animation_name=self.animation_name,
               prog_bar=prog_bar, fps=fps, clim=clim, cmap=cmap, **kwargs)

        self.remove_dir(clear)

    def build_3d_grid(self, tissue):

        if tissue.meta["type"] == "Grid":
            grid = PyVistaMeshGrid(tissue.mesh, dr=tissue.dr, as_surface=True)

        elif tissue.meta["type"] == "Elements":

            if tissue.meta["shape"] in ElementType.surface:
                grid = PyVistaSurfaceGrid(tissue.coords, tissue.elems)
            elif tissue.meta["shape"] == ElementType.TETRA:
                grid = PyVistaTetraGrid(tissue.coords, tissue.elems, as_surface=True)
            else:
                raise ValueError("Invalid element type for 3D image builder")

        else:
            raise ValueError("Invalid tissue type for 3D image builder")
                
        return grid

    def remove_dir(self, clear=True):
        if not clear or not Path(self.path, self.dir_name).exists():
            return

        import shutil
        shutil.rmtree(Path(self.path, self.dir_name))
