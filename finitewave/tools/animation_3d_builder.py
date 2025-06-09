from pathlib import Path
import numpy as np
import pyvista as pv
from natsort import natsorted
from tqdm import tqdm

from finitewave.tools.vis_mesh_builder_3d import VisMeshBuilder3D


class Animation3DBuilder:
    def __init__(self) -> None:
        pass

    def load_scalar(self, path, mask=None):
        """Load the scalar field from a file.

        Args:
            path (str): Path to the snapshot folder.
            mask (np.array, optional): Mask to apply to the scalar field.

        Returns:
            np.array: Scalar field.
        """

        scalar = np.load(path).astype(float)

        if mask is None:
            return scalar

        if mask.shape == scalar.shape:
            return scalar

        if mask[mask > 0].shape == scalar.shape:
            scalar_mesh = np.zeros_like(mask, dtype=float)
            scalar_mesh[mask > 0] = scalar
            return scalar_mesh

        raise ValueError("Mask and scalar must have the same shape, or scalar"
                         + " must have the same shape as mask[mask > 0]")

    def write(self, path, mask=None, path_save=None, window_size=(800, 800),
              clim=[0, 1], scalar_name="Scalar", animation_name="animation",
              cmap="viridis", scalar_bar=False, format="mp4", prog_bar=True,
              **kwargs):
        """Write the animation to a file.

        Args:
            path (str): Path to the snapshot folder.
            mask (np.array, optional): Mask to apply to the scalar field.
                Defaults to None.
            path_save (str, optional): Path to save the animation.
                Defaults is parent directory of path.
            window_size (tuple, optional): Size of the window.
                Defaults to (800, 800).
            clim (list, optional): Color limits. Defaults to [0, 1].
            scalar_name (str, optional): Name of the scalar field.
                Defaults to "Scalar".
            cmap (str, optional): Color map. Defaults to "viridis".
            scalar_bar (bool, optional): Show scalar bar. Defaults to False.
            format (str, optional): Format of the animation. Defaults to "mp4".
                Other options are "gif".
        """

        files = natsorted(Path(path).glob("*.npy"))

        if len(files) == 0:
            raise ValueError("No files found")

        if path_save is None:
            path_save = path.parent

        scalar = self.load_scalar(files[0], mask)

        if mask is None:
            mask = np.ones_like(scalar)

        mesh_builder = VisMeshBuilder3D()
        mesh_builder.build_mesh(mask)

        pl = pv.Plotter(notebook=False, off_screen=True,
                        window_size=window_size)

        if format == "mp4":
            pl.open_movie(Path(path_save).joinpath(f'{animation_name}.mp4'),
                          **kwargs)
        elif format == "gif":
            pl.open_gif(str(Path(path_save).joinpath(f'{animation_name}.gif')),
                        **kwargs)
        else:
            raise ValueError("Format must be 'mp4' or 'gif'")

        mesh_builder.add_scalar(scalar, scalar_name)
        pl.add_mesh(mesh_builder.grid, scalars=scalar_name,
                    clim=clim, cmap=cmap, show_scalar_bar=scalar_bar)

        pl.show(auto_close=False)

        pl.write_frame()

        for filename in tqdm(files[1:], disable=not prog_bar,
                             desc="Building animation"):
            scalar = self.load_scalar(filename, mask)
            mesh_builder.add_scalar(scalar, scalar_name)
            pl.write_frame()

        pl.close()
