from finitewave.core.diffusion.diffusion_model_base import DiffusionModelBase

from .stencil.isotropic_stencil import IsotropicStencil
from .stencil.anisotropic_stencil import AnisotropicStencil
from .assembler.elements.triangle_element import LinearTriangleElement
from .assembler.elements.quadrilateral_element import LinearQuadrilateralElement
from .assembler.elements.tetrahedral_element import LinearTetrahedralElement
from .assembler.surface_assembler import SurfaceAssembler
from .assembler.volume_assembler import VolumeAssembler


class DiffusionModel(DiffusionModelBase):
    def __init__(self):
        super().__init__()
        self.assembler = None
        self.simulation = None

    def initialize(self, simulation):
        super().initialize(simulation)
        self.simulation = simulation

        if self.assembler is None:
            self.assembler = self.default_assembler()

        self.compute_weights()

    def compute_weights(self):
        self.weights = self.assembler.assemble_matrices(self.simulation)

    def default_assembler(self):
        if self.simulation.cardiac_tissue.meta['type'] == 'Grid':
            return self._default_stencil()

        if self.simulation.cardiac_tissue.meta['type'] == 'Elements':
            return self._default_assembler()

    def _default_stencil(self):
        if self.simulation.cardiac_tissue.fibers is None:
            return IsotropicStencil()

        return AnisotropicStencil()

    def _default_assembler(self):
        if self.simulation.cardiac_tissue.meta['shape'] == 'Triangle':
            assembler = SurfaceAssembler()
            assembler.reference_element = LinearTriangleElement()
            return assembler

        if self.simulation.cardiac_tissue.meta['shape'] == 'Quadrilateral':
            assembler = SurfaceAssembler()
            assembler.reference_element = LinearQuadrilateralElement()
            return assembler

        if self.simulation.cardiac_tissue.meta['shape'] == 'Tetrahedral':
            assembler = VolumeAssembler()
            assembler.reference_element = LinearTetrahedralElement()
            return assembler

        raise ValueError("Unknown element type")
