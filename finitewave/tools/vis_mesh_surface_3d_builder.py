import pyvista as pv
import numpy as np


class VisMeshSurface3DBuilder:
    """Class to build a 3D mesh for visualization with pyvista.

    Attributes:
    ------------
    surf : pv.UnstructuredGrid
        Masked grid with cells on the surface of the mesh > 0.

    full_grid : pv.ImageData
        Full grid with all cells.
    """
    def __init__(self):
        self.surf = None
        self.full_grid = None

    def build_mesh(self, mesh):
        """Build a Unstructured Grid from 3D mesh where mesh > 0.

        Parameters:
        ------------
        mesh : np.array
            3D mesh with cardiomyocytes (elems = 1), empty space (elems = 0),
            and fibrosis (elems = 2).

        Returns:
        ------------
        surf : pv.UnstructuredGrid
            Masked grid with cells where mesh > 0.
        """
        self._mesh = mesh
        grid = pv.ImageData()
        grid.dimensions = np.array(mesh.shape) + 1
        grid.spacing = (1, 1, 1)
        grid.cell_data['mesh'] = mesh.astype(float).flatten(order='F')
        grid.cell_data['idx'] = np.arange(mesh.size)

        self.full_grid = grid
        # Threshold the mesh to remove empty space
        grid_thresh = grid.threshold(0.5, scalars='mesh')

        # Extract the surface
        self.surf = grid_thresh.extract_surface()
        indices = self.surf.cell_data['idx']
        self.indices = np.unravel_index(indices, mesh.shape, order='F')
        return self.surf

    def add_scalar(self, scalars, name='Scalars'):
        """
        Add a scalar field to the mesh. The scalar field is flattened
        and only the values of the non-empty space are added to the mesh.

        Parameters
        ----------
        scalars : np.array
            3D scalar field.
        name : str, optional
            Name of the scalar. Default is 'Scalars'.

        Returns
        -------
        surf : pv.UnstructuredGrid
            Grid with the scalar field added.
        """

        if scalars.shape != self._mesh.shape:
            raise ValueError("Scalars must have the same shape as the mesh.")

        self.surf.cell_data[name] = scalars[self.indices]
        self.surf.set_active_scalars(name)
        return self.surf
