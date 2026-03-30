import pyvista as pv
import numpy as np


class PyVistaGridBuilder:
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

    def build_from_surface_elems(self, coords, elems):
        """Build a Unstructured Grid from coords and elems.

        Parameters:
        ------------
        coords : np.array
            Coordinates of the mesh nodes.
        elems : np.array
            Elements of the mesh.

        as_surface : bool, optional
            If True, build a surface mesh. Default is False.

        Returns:
        ------------
        grid : pv.UnstructuredGrid
            Masked grid with cells where mesh > 0.
        """
        faces = np.hstack([[elems.shape[1], *elem] for elem in elems])
        self.grid = pv.PolyData(coords, faces)
        self.indices = np.arange(elems.shape[0])
        return self.grid

    def build_from_tetrahedra(self, coords, elems, as_surface=False):
        """Build a Unstructured Grid from coords and elems.

        Parameters:
        ------------
        coords : np.array
            Coordinates of the mesh nodes.
        elems : np.array
            Elements of the mesh.

        as_surface : bool, optional
            If True, build a surface mesh. Default is False.

        Returns:
        ------------
        grid : pv.UnstructuredGrid
            Masked grid with cells where mesh > 0.
        """
        faces = np.hstack([[elems.shape[1], *elem] for elem in elems])
        cell_types = np.full(elems.shape[0], pv.VTK_TETRA)
        grid = pv.UnstructuredGrid(faces, cell_types, coords)

        if as_surface:
            grid = grid.extract_surface(algorithm="geometry")

        self.grid = grid
        self.indices = self.grid.cell_data['idx']   

        return grid

    def build_from_grid(self, mesh, as_surface=False):
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
            self.grid = self.grid.extract_surface(algorithm="geometry")

        self.indices = np.unravel_index(self.grid.cell_data['idx'],
                                        mesh.shape, order='F')
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
        self.grid.cell_data[name] = scalars[self.indices]
        self.grid.set_active_scalars(name)
        return self.grid
    
    def add_masked_scalar(self, scalars, mask, name='Scalars'):
        """Add a scalar field to the mesh. The scalars assumed to be
        values where mask is True.

        Parameters
        ----------
        scalars : np.array
            Flat scalar field.
        mask : np.array
            Boolean mask where scalars are defined.
        name : str, optional
            Name of the scalar. Default is 'Scalars'.

        Returns
        -------
        grid : pv.UnstructuredGrid
            Grid with the scalar field added as active cell scalars.
        """
        scalars_mesh = np.zeros_like(mask, dtype=scalars.dtype)
        scalars_mesh[mask] = scalars
        self.grid.cell_data[name] = scalars_mesh[self.indices]
        self.grid.set_active_scalars(name)
        return self.grid

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

        if vectors.shape[:3] != self._mesh.shape:
            raise ValueError("Vectors must have the same shape as the mesh.")

        self.grid.cell_data[name] = vectors[self.indices[0], self.indices[1],
                                            self.indices[2], :]
        self.grid.set_active_vectors(name)
        return self.grid
