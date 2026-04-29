from pathlib import Path
import numpy as np
from tqdm import tqdm

from finitewave.tools.vis_mesh_builder_3d import VisMeshBuilder3D


def _require_3d_tools(format: str):
    # natsort is required for correct temporal ordering
    try:
        from natsort import natsorted
    except ImportError as e:
        raise ImportError(
            "3D animation requires natural sorting of frame files.\n"
            "Install optional dependencies with:\n"
            "  pip install natsort"
        ) from e

    # pyvista is required
    try:
        import pyvista as pv
    except ImportError as e:
        raise ImportError(
            "3D animation requires PyVista.\n"
            "Install optional dependencies with:\n"
            "  pip install pyvista"
        ) from e

    # Optional but highly recommended: explicit check for ffmpeg when writing mp4
    # because many PyVista movie pipelines rely on system FFmpeg.
    if format == "mp4":
        try:
            import imageio_ffmpeg 
        except ImportError as e:
            raise ImportError(
                "MP4 export requires imageio-ffmpeg.\n"
                "Install optional dependencies with:\n"
                "  pip install imageio-ffmpeg"
            ) from e

    return natsorted, pv


class Animation3DBuilder:
    def __init__(self) -> None:
        pass

    def load_scalar(self, path, mask=None):
        scalar = np.load(path).astype(float)

        if mask is None:
            return scalar

        if mask.shape == scalar.shape:
            return scalar

        if mask[mask > 0].shape == scalar.shape:
            scalar_mesh = np.zeros_like(mask, dtype=float)
            scalar_mesh[mask > 0] = scalar
            return scalar_mesh

        raise ValueError(
            "Mask and scalar must have the same shape, or scalar must have the same "
            "shape as mask[mask > 0]"
        )

    def write(
        self,
        path,
        path_save=None,
        animation_name="animation",
        mask=None,
        window_size=(800, 800),
        clim=(0, 1),
        scalar_name="Scalar",
        cmap="viridis",
        scalar_bar=False,
        format="mp4",
        prog_bar=True,
        **kwargs
    ):
        """Write the animation to a file.

        Args:
            path (str): Path to the snapshot folder.
            path_save (str, optional): Path to save the animation.
                Defaults is parent directory of path.
            animation_name (str, optional): Name of the animation.
                Defaults to "animation".
            mask (np.array, optional): Mask to apply to the scalar field.
                Defaults to None.
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

        natsorted, pv = _require_3d_tools(format=format)

        path = Path(path)
        files = list(natsorted(path.glob("*.npy")))

        if not files:
            raise FileNotFoundError(f"No .npy files found in: {path}")

        if path_save is None:
            path_save = path.parent
        path_save = Path(path_save)

        scalar0 = self.load_scalar(files[0], mask)

        if mask is None:
            mask = np.ones_like(scalar0)

        mesh_builder = VisMeshBuilder3D()
        mesh_builder.build_mesh(mask)

        pl = pv.Plotter(notebook=False, off_screen=True, window_size=window_size)

        if format == "mp4":
            pl.open_movie(path_save / f"{animation_name}.mp4", **kwargs)
        elif format == "gif":
            pl.open_gif(str(path_save / f"{animation_name}.gif"), **kwargs)
        else:
            raise ValueError("format must be 'mp4' or 'gif'")

        # initial frame
        mesh_builder.add_scalar(scalar0, scalar_name)
        pl.add_mesh(
            mesh_builder.grid,
            scalars=scalar_name,
            clim=clim,
            cmap=cmap,
            show_scalar_bar=scalar_bar
        )

        pl.show(auto_close=False)
        pl.write_frame()

        for filename in tqdm(files[1:], disable=not prog_bar, desc="Building animation"):
            scalar = self.load_scalar(filename, mask)
            mesh_builder.add_scalar(scalar, scalar_name)
            pl.write_frame()

        pl.close()