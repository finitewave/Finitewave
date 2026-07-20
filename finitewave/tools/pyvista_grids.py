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
    def __init__(self, coords, elems, *args, **kwargs):
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
        super().__init__(coords, faces, *args, **kwargs)


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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mesh = None
        self.as_surface = False
        self.threshold = 0.5
        self.n_nonzero_cells = None
    
    @classmethod
    def from_mesh(cls, mesh, dr=1, as_surface=False, threshold=0.5,
                  dx=None, dy=None, dz=None):
        """Build a Unstructured Grid from 3D mesh where mesh > 0.

        Parameters:
        ------------
        mesh : np.array
            3D mesh with cardiomyocytes (elems = 1), empty space (elems = 0),
            and fibrosis (elems = 2).
        dr : float, optional
            Spacing in the mesh. Default is 1.
        as_surface : bool, optional
            If True, build a surface mesh. Default is False.
        threshold : float, optional
            Threshold value for the mesh. Default is 0.5.
        dx : float, optional
            Spacing in the x-direction. Default is None.
        dy : float, optional
            Spacing in the y-direction. Default is None.
        dz : float, optional
            Spacing in the z-direction. Default is None.
        """

        mesh = np.atleast_3d(mesh)
        grid = cls._build_grid(mesh, dr=dr, dx=dx, dy=dy, dz=dz)
        grid = cls._apply_threshold(grid, mesh, threshold)
        n_nonzero_cells = grid.n_cells
        
        if as_surface:
            grid = grid.extract_surface(algorithm="geometry")
 
        instance = cls(grid)
        instance.mesh = mesh
        instance.as_surface = as_surface
        instance.threshold = threshold
        instance.n_nonzero_cells = n_nonzero_cells

        return instance
    
    @staticmethod
    def _build_grid(mesh, dr=1, dx=None, dy=None, dz=None):
        if dx is None or dy is None or dz is None:
            dx = dy = dz = dr

        grid = pv.ImageData()
        grid.dimensions = np.array(mesh.shape) + 1
        grid.spacing = (dx, dy, dz)
        grid.cell_data['mesh'] = mesh.astype(float).flatten(order='F')
        c_inds = np.arange(mesh.size).reshape(mesh.shape)
        grid.cell_data['mesh_inds'] = c_inds.flatten(order='F')
        return grid
    
    @staticmethod
    def _apply_threshold(grid, mesh, threshold=0.5):
        grid = grid.threshold(threshold, scalars='mesh', invert=False)
        c_mesh_inds = - np.ones(mesh.shape, dtype=int)
        c_mesh_inds[mesh > threshold] = np.arange(grid.n_cells)
        non_zero_inds = np.unravel_index(grid.cell_data['mesh_inds'], mesh.shape, order='C')
        grid.cell_data['nonzero_inds'] = c_mesh_inds[*non_zero_inds]
        return grid
    
    def __setitem__(self, key, value):
        if self.mesh is None or "nonzero_inds" not in self.cell_data:
            super().__setitem__(key, value)
            return

        value = np.asarray(value)

        if value.shape[:3] == self.mesh.shape:
            inds = np.unravel_index(self.cell_data['mesh_inds'], self.mesh.shape, order='C')
            self.cell_data[key] = value[*inds, ...]
        elif value.shape[0] == self.mesh.size:
            self.cell_data[key] = value[self.cell_data['mesh_inds'], ...]
        elif value.shape[0] == self.n_nonzero_cells:
            self.cell_data[key] = value[self.cell_data['nonzero_inds'], ...]
        else:
            super().__setitem__(key, value)
