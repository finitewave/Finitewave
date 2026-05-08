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
        self.n_full_points = grid.n_points
        self.n_full_cells = grid.n_cells
        
        self.as_surface = as_surface

        if as_surface:
            grid = grid.extract_surface(algorithm="geometry")

        self.point_idx = grid.point_data['point_idx']
        self.cell_idx = grid.cell_data['cell_idx']
        super().__init__(grid)

    def __setitem__(self, name, value):
        if self.as_surface and value.shape[0] == self.n_full_points:
            self.point_data[name] = value[self.point_idx, ...]
            return
        
        if self.as_surface and value.shape[0] == self.n_full_cells:
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
    def __init__(self, mesh, dr=1, as_surface=False, threshold=0.5):
        """Build a Unstructured Grid from 3D mesh where mesh > 0.

        Parameters:
        ------------
        mesh : np.array
            3D mesh with cardiomyocytes (elems = 1), empty space (elems = 0),
            and fibrosis (elems = 2).

        as_surface : bool, optional
            If True, build a surface mesh. Default is False.
        threshold : float, optional
            Threshold value for the mesh. Default is 0.5.
        """
        self._mesh = mesh
        self.as_surface = as_surface
        self.threshold = threshold

        mesh = np.atleast_3d(mesh)
        grid = self._build_grid(mesh, dr=dr)
        self._mesh_size = grid.n_cells
        grid = self._apply_threshold(grid, mesh, threshold)
        
        self._mesh_nonzero_size = grid.n_cells

        if as_surface:
            grid = grid.extract_surface(algorithm="geometry")

        super().__init__(grid)

    def _build_grid(self, mesh, dr=1):
        grid = pv.ImageData()
        grid.dimensions = np.array(mesh.shape) + 1
        grid.spacing = tuple([dr] * 3)
        grid.cell_data['mesh'] = mesh.astype(float).flatten(order='F')
        c_inds = np.arange(mesh.size).reshape(mesh.shape)
        grid.cell_data['mesh_inds'] = c_inds.flatten(order='F')
        return grid
    
    def _apply_threshold(self, grid, mesh, threshold=0.5):
        grid = grid.threshold(threshold, scalars='mesh', invert=False)
        c_mesh_inds = - np.ones(mesh.shape, dtype=int)
        c_mesh_inds[mesh > threshold] = np.arange(grid.n_cells)
        non_zero_inds = np.unravel_index(grid.cell_data['mesh_inds'], mesh.shape, order='C')
        grid.cell_data['nonzero_inds'] = c_mesh_inds[*non_zero_inds]
        return grid
    
    def __setitem__(self, key, value):
        value = np.asarray(value)
        if value.shape[:3] == self._mesh.shape:
            inds = np.unravel_index(self.cell_data['mesh_inds'], self._mesh.shape, order='C')
            self.cell_data[key] = value[*inds, ...]
        elif value.shape[0] == self._mesh.size:
            self.cell_data[key] = value[self.cell_data['mesh_inds'], ...]
        elif value.shape[0] == self._mesh_nonzero_size:
            self.cell_data[key] = value[self.cell_data['nonzero_inds'], ...]
        else:
            super().__setitem__(key, value)
