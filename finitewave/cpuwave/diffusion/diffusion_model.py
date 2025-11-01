from finitewave.core.diffusion.diffusion_model_base import DiffusionModelBase

from .elements.triangle_element import LinearTriangleElement
from .elements.quadrilateral_element import LinearQuadrilateralElement
from .elements.tetrahedral_element import LinearTetrahedralElement

from .stencils.asymmetric_stencil import AsymmetricStencil

from .assembler.surface_assembler import SurfaceAssembler
from .assembler.volume_assembler import VolumeAssembler
from .assembler.grid_assembler import GridAssembler


class DiffusionModel(DiffusionModelBase):
    """
    Diffusion model which selects appropriate assembler based on tissue type.

    This model supports both grid-based and element-based simulations,
    adapting the diffusion stencil and assembler accordingly.

    Attributes
    ----------
    assembler : Assembler
        The assembler used to construct the diffusion matrices.
    simulation : Simulation
        The simulation instance associated with this diffusion model.
    weights : dict
        The computed diffusion weights for the simulation.
    """
    def __init__(self):
        super().__init__()
        self.assembler = None
        self.simulation = None

    def initialize(self, simulation):
        """Selects and initializes the appropriate assembler and
        computes the diffusion weights.

        Parameters
        ----------
        simulation : Simulation
            The simulation instance associated with this diffusion model.
        """
        self.simulation = simulation

        if self.assembler is None:
            self.assembler = self.default_assembler()

        self.assembler.initialize(simulation)
        self.weights = self.assembler.weights

    def compute_weights(self):
        """Computes weights using the selected assembler."""
        self.weights = self.assembler.compute_weights()

    def default_assembler(self):
        """Selects the default assembler based on tissue type.

        Returns
        -------
        Assembler
            The selected assembler instance.
        """
        if self.simulation.cardiac_tissue.meta['type'] == 'Grid':
            return self._default_stencil()

        if self.simulation.cardiac_tissue.meta['type'] == 'Elements':
            return self._default_assembler()

    def _default_stencil(self):
        assembler = GridAssembler()
        assembler.stencil = AsymmetricStencil()
        return assembler

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
