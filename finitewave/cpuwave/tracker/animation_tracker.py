from pathlib import Path

from .frame_tracker import FrameTracker
from finitewave.tools.animation_builder import (
    AnimationBuilder,
    Image2DBuilder,
    Image3DBuilder
)


class AnimationTracker(FrameTracker):
    """
    A class to track and save frames of a 2D cardiac tissue simulation simulation
    for animation purposes.

    This tracker periodically saves the state of a specified target array from
    the simulation to disk as NumPy files, which can later be used to create
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def animation_name(self):
        return self.dir_name

    @animation_name.setter
    def animation_name(self, animation_name):
        self.dir_name = animation_name

    def write(self, prog_bar=True, fps=12, clim=[0, 1], cmap="RdBu_r", clear=False, **kwargs):
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
        clear : bool, optional
            Whether to clear the saved frames after creating the animation.
        **kwargs : dict
            Additional keyword arguments for the animation builder.
        """
        if self.aggregate:
            images = self.output
        else:
            images = None

        if self.simulation.cardiac_tissue.mesh.ndim == 2:
            image_builder = Image2DBuilder()
            image_builder.build_from_tissue(self.simulation.cardiac_tissue,
                                            restore_input=not self.keep_shape,
                                            clim=clim, cmap=cmap, **kwargs)
            image_builder.collect_images(path=Path(self.path, self.dir_name),
                                         images=images)
            
        else:
            image_builder = Image3DBuilder()
            image_builder.build_from_tissue(self.simulation.cardiac_tissue,
                                            scalars=self.var_name,
                                            clim=clim, cmap=cmap, **kwargs)
            image_builder.collect_images(path=Path(self.path, self.dir_name),
                                         images=images)
        
        animation_builder = AnimationBuilder()
        animation_builder.image_builder = image_builder

        animation_builder.write(
            path=self.path,
            animation_name=self.animation_name,
            prog_bar=prog_bar,
            fps=fps,
        )

        self.remove_dir(clear)

    def remove_dir(self, clear=True):
        if not clear or not Path(self.path, self.dir_name).exists():
            return

        import shutil
        shutil.rmtree(Path(self.path, self.dir_name))
