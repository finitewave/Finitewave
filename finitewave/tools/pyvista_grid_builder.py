import pyvista as pv
import numpy as np


class PyVistaSurfaceGrid(pv.PolyData):
    """Class to hold a pyvista surface grid and its associated data.

    Attributes:
    ------------
    grid : pv.PolyData
        Masked grid with cells where mesh > 0.
    indices : tuple of np.array
        Indices of the non-empty cells in the original mesh.
    """
    def __init__(self, coords, elems):
        """Build a PolyData from coords and elems.

        Parameters:
        ------------
        coords : np.array
            Coordinates of the mesh nodes.
        elems : np.array
            Elements of the mesh.
        """
        if coords.shape[1] == 2:
            coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])

        faces = np.hstack([np.full((elems.shape[0], 1), elems.shape[1]), elems]).ravel()
        super().__init__(coords, faces)


class PyVistaTetraGrid(pv.UnstructuredGrid):
    """Class to hold a pyvista tetrahedral grid and its associated data.

    Attributes:
    ------------
    grid : pv.UnstructuredGrid
        Masked grid with cells where mesh > 0.
    indices : tuple of np.array
        Indices of the non-empty cells in the original mesh.
    """
    def __init__(self, coords, elems, as_surface=False):
        """Build a Unstructured Grid from coords and elems.

        Parameters:
        ------------
        coords : np.array
            Coordinates of the mesh nodes.
        elems : np.array
            Elements of the mesh.

        as_surface : bool, optional
            If True, build a surface mesh. Default is False.
        """
        cells = np.hstack([np.full((elems.shape[0], 1), 4), elems]).ravel()
        celltypes = np.full(elems.shape[0], pv.CellType.TETRA, dtype=np.uint8)

        grid = pv.UnstructuredGrid(cells, celltypes, coords)
        grid.point_data['point_idx'] = np.arange(coords.shape[0])
        grid.cell_data['cell_idx'] = np.arange(elems.shape[0])
        
        self.as_surface = as_surface
        if as_surface:
            grid = grid.extract_surface(algorithm="geometry")

        self.point_idx = grid.point_data['point_idx']
        self.cell_idx = grid.cell_data['cell_idx']
        super().__init__(grid)

    def __setitem__(self, name, value):
        if self.as_surface and value.shape[0] == self.n_points:
            self.point_data[name] = value[self.point_idx, ...]
            return
        
        if self.as_surface and value.shape[0] == self.n_cells:
            self.cell_data[name] = value[self.cell_idx, ...]
            return
        
        super().__setitem__(name, value)


class PyVistaMeshGrid(pv.UnstructuredGrid):
    """Class to hold a pyvista grid and its associated data.

    Attributes:
    ------------
    grid : pv.UnstructuredGrid
        Masked grid with cells where mesh > 0.
    indices : tuple of np.array
        Indices of the non-empty cells in the original mesh.
    """
    def __init__(self, mesh, dr=1, as_surface=False):
        """Build a Unstructured Grid from 3D mesh where mesh > 0.

        Parameters:
        ------------
        mesh : np.array
            3D mesh with cardiomyocytes (elems = 1), empty space (elems = 0),
            and fibrosis (elems = 2).

        as_surface : bool, optional
            If True, build a surface mesh. Default is False.
        """
        self._mesh = mesh
        shape = mesh.shape[::-1]

        if mesh.ndim == 2:
            shape = (shape[0], shape[1], 1)

        grid = pv.ImageData()
        grid.dimensions = np.array(shape) + 1
        grid.spacing = tuple([dr] * 3)
        grid.cell_data['mesh'] = mesh.astype(float).flatten(order='C')
        grid.cell_data['full_idx'] = np.arange(mesh.size)
        self.n_full_cells = grid.n_cells

        # Threshold the mesh to remove empty cells
        grid = grid.threshold(0.5)
        self.n_grid_cells = grid.n_cells
        grid.cell_data["grid_idx"] = np.arange(self.n_grid_cells)
        
        self.as_surface = as_surface
    
        if as_surface:
            grid = grid.extract_surface(algorithm="geometry")

        self.n_surface_cells = grid.n_cells

        # self.indices = np.unravel_index(grid.cell_data['idx'], mesh.shape, order='F')
        super().__init__(grid)
    
    def __setitem__(self, key, value):
        value = np.asarray(value)
        if value.shape[:3] == self._mesh.shape:
            inds = np.unravel_index(self.cell_data['full_idx'], self._mesh.shape, order='C')
            self.cell_data[key] = value[*inds, ...]
        elif value.shape[0] == self.n_full_cells:
            self.cell_data[key] = value[self.cell_data['full_idx'], ...]
        elif value.shape[0] == self.n_grid_cells:
            self.cell_data[key] = value[self.cell_data['grid_idx'], ...]
        else:
            super().__setitem__(key, value)
                    
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
        self.cell_data[name] = scalars_mesh[self.indices]
