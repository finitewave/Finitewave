from pathlib import Path
import shutil as shatilib

from finitewave.cpuwave2D.tracker.animation_2d_tracker import (
    Animation2DTracker
)
from finitewave.tools.animation_3d_builder import Animation3DBuilder


class Animation3DTracker(Animation2DTracker):
    """A class to track and save frames of a 3D cardiac tissue model simulation
    for animation purposes.
    """
    def __init__(self):
        """
        Initializes the Animation3DTracker with default parameters.
        """
        super().__init__()

    def write(self, path=None, clim=[0, 1], cmap="viridis", scalar_bar=False,
              format="mp4", clear=False, prog_bar=True, **kwargs):
        """
        Write the animation to a file.

        Parameters
        ----------
        path : str, optional
            Path to save the animation. Defaults to path of the tracker.
        clim : list, optional
            Color limits. Defaults to [0, 1].
        cmap : str, optional
            Color map. Defaults to "viridis".
        scalar_bar : bool, optional
            Show scalar bar. Defaults to False.
        format : str, optional
            Format of the animation. Defaults to "mp4". Other option is "gif".
        clear : bool, optional
            Clear the snapshot folder after writing the animation.
            Defaults to False.
        **kwargs : optional
            Additional arguments for the animation writer.
        """

        if path is None:
            path = self.path

        animation_builder = Animation3DBuilder()
        animation_builder.write(Path(self.path, self.dir_name),
                                path_save=path,
                                mask=self.model.cardiac_tissue.mesh,
                                scalar_name=self.variable_name,
                                clim=clim, cmap=cmap,
                                scalar_bar=scalar_bar, format=format,
                                prog_bar=prog_bar, **kwargs)

        if clear:
            shatilib.rmtree(Path(self.path, self.dir_name))
