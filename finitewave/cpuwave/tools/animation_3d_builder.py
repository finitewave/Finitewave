from pathlib import Path
import numpy as np
import pyvista as pv
from tqdm import tqdm

from .animation_2d_builder import Animation2DBuilder
from .pyvista_grid_builder import PyVistaGridBuilder


class Animation3DBuilder(Animation2DBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.mesh_builder = PyVistaGridBuilder()

    def write(self,
              coords=None,
              elems=None,
              mesh=None,
              nan_mask=None,
              elem_type=None,
              window_size=(800, 800),
              clim=[0, 1],
              cmap="RdBu_r",
              format="mp4",
              scalar_name="Scalar",
              scalar_bar=False,
              camera_position="iso",
              **kwargs):
        """Write the animation to a file.

        Parameters
        ----------
        coords : ndarray, optional
            Coordinates of the mesh nodes.
        elems : ndarray, optional
            Elements of the mesh.
        mesh : ndarray, optional
            Mesh grid to build the visualization mesh.
        nan_mask : ndarray, optional
            Mask where to apply NaN values.
        window_size : tuple, optional
            Size of the rendering window.
        clim : list, optional
            Color limits for the scalar data.
        cmap : str, optional
            Colormap to use for rendering.
        format : str, optional
            Output format ('mp4' or 'gif').
        scalar_name : str, optional
            Name of the scalar data.
        scalar_bar : bool, optional
            Whether to show the scalar bar.
        camera_position : str or list, optional
            Camera position for rendering.
        **kwargs : dict
            Additional keyword arguments for the movie/gif writer.
        """

        files = self.collect_frames(self.path)[:-10:2]
        path_save = self.make_path_save(format)

        scalar = self.load_scalar(files[0], self.scalar_mask, nan_mask)

        grid = self.build_grid(coords, elems, mesh, elem_type)
        scalar = self.calc_cell_scalars(scalar, elems, mesh)
        grid = self.mesh_builder.add_scalar(scalar, scalar_name)

        pl = pv.Plotter(notebook=False, off_screen=True, window_size=window_size)
        pl.add_mesh(grid, cmap=cmap, show_edges=False, clim=clim,
                    scalars=scalar_name, show_scalar_bar=scalar_bar)

        if format == "mp4":
            print(f"Saving animation to {path_save}")
            pl.open_movie(path_save, format='FFMPEG', **kwargs)
        elif format == "gif":
            pl.open_gif(path_save, **kwargs)
        else:
            raise ValueError("Format must be 'mp4' or 'gif'")

        pl.camera_position = camera_position
        pl.show(auto_close=False)
        pl.write_frame()

        for filename in tqdm(files[1:], disable=not self.prog_bar,
                             desc="Building animation"):
            scalar = self.load_scalar(filename, self.scalar_mask, nan_mask)
            scalar = self.calc_cell_scalars(scalar, elems, mesh)
            grid = self.mesh_builder.add_scalar(scalar, scalar_name)
            pl.write_frame()

        pl.close()

    def make_path_save(self, format):
        if self.path_save is None:
            self.path_save = Path(self.path).parent

        return Path(self.path_save).joinpath(f'{self.animation_name}.{format}')

    def calc_cell_scalars(self, scalar, elems, mesh):
        if mesh is not None:
            return scalar

        return np.mean(scalar[elems], axis=1)

    def build_grid(self, coords, elems, mesh, elem_type="Tri"):
        """Build a PyVista mesh from coordinates and elements.

        Parameters
        ----------
        coords : ndarray, optional
            Coordinates of the mesh nodes.
        elems : ndarray, optional
            Elements of the mesh.
        mesh : ndarray, optional
            Mesh grid to build the visualization mesh.

        Returns
        -------
        pv.PolyData
            The constructed PyVista mesh.
        """
        if elems is None and mesh is None:
            raise ValueError("Either elems or mesh must be provided.")

        if elems is not None and mesh is not None:
            raise ValueError("Only one of elems or mesh should be provided.")

        if mesh is not None:
            self.mesh_builder.build_from_grid(mesh, as_surface=True)
            return self.mesh_builder.grid

        if elem_type is None:
            raise ValueError("elem_type must be specified when elems are provided.")

        if 'Tetra' in elem_type:
            self.mesh_builder.build_from_tetrahedra(coords, elems,
                                                    as_surface=True)
            return self.mesh_builder.grid

        self.mesh_builder.build_from_surface_elems(coords, elems)
        return self.mesh_builder.grid
