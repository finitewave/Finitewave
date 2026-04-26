from pathlib import Path
from tqdm import tqdm
import natsort
import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np
import av

from .pyvista_grid_builder import PyVistaMeshGrid, PyVistaSurfaceGrid, PyVistaTetraGrid
from finitewave.core.numerics.fem.elements.element_type import ElementType


class AnimationBuilder:
    def __init__(self):
        self.image_builder = None

    def write(self, path=".", animation_name="frames", prog_bar=False, fps=24):
        
        path = Path(path, animation_name)

        container = av.open(str(path.with_suffix(".mp4")), mode="w")
        
        stream = container.add_stream("libx264", rate=fps)
        stream.width = self.image_builder.width
        stream.height = self.image_builder.height
        stream.pix_fmt = "yuv444p"

        for frame_i in tqdm(range(self.image_builder.total_frames),
                            desc="Building animation", disable=not prog_bar):
            # Grab the frame as a NumPy RGB array
            img = self.image_builder.generate_image(frame_i)
            
            # Convert to PyAV frame and mux
            frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)

        # Cleanup
        for packet in stream.encode():
            container.mux(packet)

        container.close()
        self.image_builder.finalize()


class Image2DBuilder:
    def __init__(self):
        self.images = None
        self.files = None
        self.total_frames = None
        self.output_mask = None
        self.upscale_factor = 1

    def collect_images(self, path=None, images=None):
        if images is not None:
            total_frames = len(images)
            self.images = images
        else:
            self.files, total_frames = self.collect_files(path)

        self.total_frames = total_frames

    def build_from_tissue(self, cardiac_tissue, restore_input=False, upscale_factor=1,
                          clim=[0, 1], cmap="viridis", **kwargs):
        self.build_grid(cardiac_tissue.mesh, restore_input, upscale_factor)
        self.cmap = self.setup_cmap(cmap)
        self.clim = clim

    def build_grid(self, mesh, restore_input=False, upscale_factor=1):
        if restore_input:
            self.output_mask = mesh > 0

        self.upscale_factor = upscale_factor

        self.width = mesh.shape[1] * upscale_factor
        self.height = mesh.shape[0] * upscale_factor

    def collect_files(self, path):
        files = natsort.natsorted(Path(path).glob("*.npy"))
        if len(files) == 0:
            raise ValueError(f"No files found in {Path(path)}")
        total_frames = len(files)
        return files, total_frames
    
    def generate_image(self, img_id):
        if self.images is not None:
            img = self.images[img_id]
        else:
            img = self.load_frame(self.files, img_id)
        
        img = self.filter_frame(img, self.output_mask, self.upscale_factor)
        img = self.torgb_frame(img, self.clim, self.cmap)
        return img

    def load_frame(self, files, frame_i):
        img = np.load(files[frame_i])
        return img

    def filter_frame(self, frame, output_mask=None, upscale_factor=1):
        output = frame

        if output_mask is not None:
            output = np.full_like(output_mask, fill_value=np.nan, dtype=frame.dtype)
            output[output_mask] = frame

        if upscale_factor > 1:
            output = np.repeat(output, upscale_factor, axis=0)
            output = np.repeat(output, upscale_factor, axis=1)

        return output

    def torgb_frame(self, frame, clim, cmap):
        mask = (frame < clim[0]) | (frame > clim[1])
        frame = (frame - clim[0]) / (clim[1] - clim[0])
        frame[mask] = np.nan
        frame = (cmap(frame, bytes=True)[:, :, :3]).astype("uint8")
        return frame

    def setup_cmap(self, cmap, nan_color="black"):
        cmap = plt.get_cmap(cmap)
        cmap.set_bad(color=nan_color)
        return cmap
    
    def finalize(self):
        pass


class Image3DBuilder(Image2DBuilder):
    def __init__(self):
        super().__init__()
        self.images = None
        self.files = None
        self.total_frames = None

    def collect_images(self, path, images=None):
        if images is not None:
            total_frames = len(images)
            self.images = images
        else:
            self.files, total_frames = self.collect_files(path)
        
        self.total_frames = total_frames

    def generate_image(self, img_id):
        if self.images is not None:
            img = self.images[img_id]
        else:
            img = self.load_frame(self.files, img_id)
        
        self.grid[self.scalars] = img
        self.plotter.render()
        img = self.plotter.screenshot(None) 
        return img

    def build_from_tissue(self, cardiac_tissue, camera_position='iso', scalars="u",
                          clim=[0, 1], cmap="viridis", window_size=(800, 800), **kwargs):
        
        self.width, self.height = window_size
        self.scalars = scalars

        if cardiac_tissue.meta["type"] == "Grid":
            self.grid = self._build_from_mesh(cardiac_tissue.mesh)
        
        elif cardiac_tissue.meta["type"] == "Elements":
            self.grid = self._build_from_elements(cardiac_tissue.coords, cardiac_tissue.elems,
                                            cardiac_tissue.meta["shape"])
        else:
            raise ValueError("Invalid tissue type")
        
        self.grid[self.scalars] = np.zeros_like(cardiac_tissue.mesh, dtype=np.float32)

        self.plotter = pv.Plotter(off_screen=True, window_size=(self.width, self.height))
        self.plotter.add_mesh(self.grid, cmap=cmap, clim=clim, scalars=scalars, **kwargs)
        self.plotter.camera_position = camera_position
        self.plotter.add_axes(line_width=5)
        self.plotter.show(auto_close=False)

    def _build_from_mesh(self, mesh):
        return PyVistaMeshGrid(mesh, as_surface=True)

    def _build_from_elements(self, coords, elems, elem_type):
        if elem_type in ElementType.surface:
            grid = PyVistaSurfaceGrid(coords, elems)
            return grid
        
        if elem_type == ElementType.TETRA:
            grid = PyVistaTetraGrid(coords, elems)
            return grid
        
        raise ValueError("Invalid element type")
    
    def finalize(self):
        self.plotter.close()
