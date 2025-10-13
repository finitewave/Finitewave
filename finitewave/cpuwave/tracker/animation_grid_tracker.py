from pathlib import Path
import numpy as np

from .frame_grid_tracker import FrameGridTracker
from finitewave.cpuwave.tools.animation_2d_builder import Animation2DBuilder
from finitewave.cpuwave.tools.animation_3d_builder import Animation3DBuilder


class AnimationGridTracker(FrameGridTracker):
    """
    A class to track and save frames of a 2D cardiac tissue model simulation
    for animation purposes.

    This tracker periodically saves the state of a specified target array from
    the model to disk as NumPy files, which can later be used to create
    animations.

    Attributes
    ----------
    dir_name : str
        Directory for saving frames.
    variable_name : str
        Name of the target array to capture.
    frame_type : str
        Default frame format settings.
    overwrite : bool
        Overwrite existing frames.
    """

    def __init__(self):
        """
        Initializes the AnimationGridTracker with default parameters.
        """
        super().__init__()
        self.animation_name = "animation"
        self.animation_builder = None

    def initialize(self, model):
        super().initialize(model)
        if self.model.cardiac_tissue.mesh.ndim == 2:
            self.animation_builder = Animation2DBuilder()
        elif self.model.cardiac_tissue.mesh.ndim == 3:
            self.animation_builder = Animation3DBuilder()

        self.animation_builder.path = Path(self.path, self.dir_name)
        self.animation_builder.prog_bar = model.prog_bar
        self.animation_builder.animation_name = self.animation_name

        if self.mask_output:
            scalar_mask = self.model.cardiac_tissue.mesh == 1
            self.animation_builder.scalar_mask = scalar_mask

    @property
    def animation_name(self):
        return self.dir_name

    @animation_name.setter
    def animation_name(self, animation_name):
        self.dir_name = animation_name

    def write(self, path_save=None, cmap="RdBu_r", clim=[0, 1], clear=False,
              **kwargs):
        """
        Creates an animation from the saved frames using the Animation2DBuilder
        class. Fibrosis and boundaries will be shown in black.

        Parameters
        ----------
        path_save : str or Path, optional
            Path to save the animation file. If None, it will be saved in the
            `self.path`.
        shape_scale : int, optional
            Scale factor for the frame size. The default is 5.
        fps : int, optional
            Frames per second for the animation. The default is 12.
        cmap : str, optional
            Color map for the animation. The default is 'coolwarm'.
        clim : list, optional
            Color limits for the animation. The default is [0, 1].
        clear : bool, optional
            Clear the snapshot folder after creating the animation.
            The default is False.
        """
        self.animation_builder.path_save = path_save
        self.animation_builder.write(
            mask=self.model.cardiac_tissue.mesh != 1,
            cmap=cmap,
            clim=clim,
            **kwargs
        )

        self.remove_dir(clear)

    def remove_dir(self, clear=True):
        if not clear:
            return

        import shutil
        shutil.rmtree(Path(self.path, self.dir_name))
