from pathlib import Path
import numpy as np

from .frame_tracker_fdm import FrameTrackerFDM
from finitewave.tools import Animation2DBuilder


class AnimationTrackerFDM(FrameTrackerFDM):
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
        Initializes the AnimationTrackerFDM with default parameters.
        """
        super().__init__()

    def write(self,
              path_save=None,
              animation_name="animation",
              shape_scale=1,
              fps=12,
              cmap="RdBu_r",
              clim=[0, 1],
              clear=False,
              prog_bar=True):
        """
        Creates an animation from the saved frames using the Animation2DBuilder
        class. Fibrosis and boundaries will be shown in black.

        Parameters
        ----------
        path_save : str or Path, optional
            Path to save the animation file. If None, it will be saved in the
            `self.path`.
        animation_name : str, optional
            Name of the animation file. Defaults to the directory name.
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
        prog_bar : bool, optional
            Show a progress bar during the animation creation.
            The default is True.
        """
        if self.model.cardiac_tissue.mesh.ndim == 2:
            self.write_2d(path_save=path_save,
                          animation_name=animation_name,
                          shape_scale=shape_scale,
                          fps=fps,
                          cmap=cmap,
                          clim=clim,
                          clear=clear,
                          prog_bar=prog_bar)

        if self.model.cardiac_tissue.mesh.ndim == 3:
            self.write_3d(path_save=path_save,
                          animation_name=animation_name,
                          shape_scale=shape_scale,
                          fps=fps,
                          cmap=cmap,
                          clim=clim,
                          clear=clear,
                          prog_bar=prog_bar)

    def write_3d(self,
                 path_save=None,
                 animation_name=None,
                 shape_scale=1,
                 fps=12,
                 cmap="RdBu_r",
                 clim=[0, 1],
                 clear=False,
                 prog_bar=True):
        pass

    def write_2d(
            self,
            path_save=None,
            animation_name=None,
            shape_scale=1,
            fps=12,
            cmap="RdBu_r",
            clim=[0, 1],
            clear=False,
            prog_bar=True):
        """
        Creates an animation from the saved frames using the Animation2DBuilder
        class. Fibrosis and boundaries will be shown in black.

        Parameters
        ----------
        path_save : str or Path, optional
            Path to save the animation file. If None, it will be saved in the
            `self.path`.
        animation_name : str, optional
            Name of the animation file. Defaults to the directory name.
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
        prog_bar : bool, optional
            Show a progress bar during the animation creation.
            The default is True.
        """
        animation_builder = Animation2DBuilder()
        animation_builder.path = Path(self.path, self.dir_name)
        animation_builder.prog_bar = prog_bar

        if path_save is None:
            path_save = self.path

        animation_builder.path_save = Path(path_save).resolve()

        if self.mask_output:
            animation_builder.scalar_mask = self.model.cardiac_tissue.mesh == 1

        if animation_name is None:
            animation_name = self.dir_name

        animation_builder.write(
            animation_name=animation_name,
            mask=self.model.cardiac_tissue.mesh != 1,
            shape_scale=shape_scale,
            fps=fps,
            clim=clim,
            cmap=cmap,
        )

        if clear:
            import shutil
            shutil.rmtree(Path(self.path, self.dir_name))
