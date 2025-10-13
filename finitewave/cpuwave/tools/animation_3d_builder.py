from pathlib import Path
import numpy as np
import pyvista as pv
from natsort import natsorted
from tqdm import tqdm

from .vis_mesh_builder_3d import VisMeshBuilder3D
from .animation_2d_builder import Animation2DBuilder


class Animation3DBuilder(Animation2DBuilder):
    def __init__(self) -> None:
        super().__init__()

    def write(self,
              mask=None,
              window_size=(800, 800),
              clim=[0, 1],
              cmap="viridis",
              format="mp4",
              scalar_name="Scalar",
              scalar_bar=False,
              camera_position="iso",
              **kwargs):
        """Write the animation to a file.

        Args:
            mask (np.array): Mask to apply to the scalar field.
            window_size (tuple, optional): Size of the window.
                Defaults to (800, 800).
            clim (list, optional): Color limits. Defaults to [0, 1].
            cmap (str, optional): Color map. Defaults to "viridis".
            format (str, optional): Format of the animation. Defaults to "mp4".
                Other options are "gif".
            scalar_name (str, optional): Name of the scalar field.
                Defaults to "Scalar".
            scalar_bar (bool, optional): Show scalar bar. Defaults to False.
        """

        path = Path(self.path)
        files = natsorted(path.glob("*.npy"))

        path_save = self.path_save

        if path_save is None:
            path_save = path.parent

        path_save = Path(path_save).joinpath(f'{self.animation_name}.{format}')

        scalar = self.load_scalar(files[0], self.scalar_mask)

        if self.scalar_mask is None:
            mesh = np.ones_like(scalar)
            mesh[np.isnan(scalar)] = 0
        else:
            mesh = self.scalar_mask

        if mask is not None:
            scalar[mask] = np.nan

        mesh_builder = VisMeshBuilder3D()
        mesh_builder.build_mesh(mesh, as_surface=True)
        grid = mesh_builder.add_scalar(scalar, scalar_name)

        pl = pv.Plotter(notebook=False, off_screen=True,
                        window_size=window_size)

        if format == "mp4":
            pl.open_movie(path_save, **kwargs)
        elif format == "gif":
            pl.open_gif(path_save, **kwargs)
        else:
            raise ValueError("Format must be 'mp4' or 'gif'")

        pl.add_mesh(grid, scalars=scalar_name,
                    clim=clim, cmap=cmap, show_scalar_bar=scalar_bar)
        pl.camera_position = camera_position

        pl.show(auto_close=False)

        pl.write_frame()

        for filename in tqdm(files[1:], disable=not self.prog_bar,
                             desc="Building animation"):
            scalar = self.load_scalar(filename, self.scalar_mask)
            if mask is not None:
                scalar[mask] = np.nan

            mesh_builder.add_scalar(scalar, scalar_name)
            pl.write_frame()

        pl.close()
