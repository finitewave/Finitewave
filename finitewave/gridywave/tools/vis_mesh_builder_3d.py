import pyvista as pv
import numpy as np


class VisMeshBuilder3D:
    """Class to build a 3D mesh for visualization with pyvista.

    Attributes:
    ------------
    grid : pv.UnstructuredGrid)
        Masked grid with cells where mesh > 0.

    full_grid : pv.ImageData
        Full grid with all cells.
    """
    def __init__(self):
        self.grid = None
        self.full_grid = None

    def build_mesh(self, mesh, as_surface=False):
        """Build a Unstructured Grid from 3D mesh where mesh > 0.

        Parameters:
        ------------
        mesh : np.array
            3D mesh with cardiomyocytes (elems = 1), empty space (elems = 0),
            and fibrosis (elems = 2).

        as_surface : bool, optional
            If True, build a surface mesh. Default is False.

        Returns:
        ------------
        grid : pv.UnstructuredGrid
            Masked grid with cells where mesh > 0.
        """
        grid = pv.ImageData()
        grid.dimensions = np.array(mesh.shape) + 1
        grid.spacing = (1, 1, 1)
        grid.cell_data['mesh'] = mesh.astype(float).flatten(order='F')
        grid.cell_data['idx'] = np.arange(mesh.size)

        self.full_grid = grid
        # Threshold the mesh to remove empty cells
        self.grid = grid.threshold(0.5)

        if as_surface:
            self.grid = self.grid.extract_surface()

        self.indices = np.unravel_index(self.grid.cell_data['idx'],
                                        mesh.shape, order='F')
        self._mesh = mesh
        self._scalars = np.zeros_like(self._mesh, dtype=float)

        return self.grid

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
        grid : pv.UnstructuredGrid
            Grid with the scalar field added.
        """

        if scalars.shape != self._mesh.shape:
            raise ValueError("Scalars must have the same shape asthe mesh.")

        self.grid.cell_data[name] = scalars[self.indices]
        self.grid.set_active_scalars(name)
        return self.grid

    def add_masked_scalar(self, scalars, name='Scalars'):
        """
        Add scalars to the mesh, where mesh > 0.

        Parameters
        ----------
        scalars : np.array
            3D scalar field.
        name : str, optional
            Name of the scalar. Default is 'Scalars'.
        """
        self._scalars[self._mesh > 0] = scalars
        return self.add_scalar(self._scalars, name)

    def add_vector(self, vectors, name='Vectors'):
        """
        Add a vector field to the mesh. The vector field is flattened
        and only the values of the non-empty space are added to the mesh.

        Parameters
        ----------
        vectors : np.array
            3D vector field.
        name : str, optional
            Name of the vector. Default is 'Vectors'.

        Returns
        -------
        grid : pv.UnstructuredGrid
            Grid with the vector field added.
        """

        if vectors.shape != self._mesh.shape + (3,):
            raise ValueError("Vectors must have the same shape as the mesh.")

        self.grid.cell_data[name] = vectors[self.indices[0], self.indices[1],
                                            self.indices[2], :]
        self.grid.set_active_vectors(name)
        return self.grid
