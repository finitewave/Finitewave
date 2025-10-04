from finitewave.gridywave.cpuwave2D.diffusion.diffusion_model_2d import (
    DiffusionModel2D
)

from .stencil.isotropic_stencil_3d import IsotropicStencil3D
from .stencil.asymmetric_stencil_3d import AsymmetricStencil3D


class DiffusionModel3D(DiffusionModel2D):
    """
    This class implements the diffusion model for 3D cardiac tissue.
    """
    def __init__(self):
        super().__init__()

    def default_stencil(self):
        """
        Returns the default stencil for 3D cardiac tissue.
        """
        if self.simulation.cardiac_tissue.fibers is None:
            return IsotropicStencil3D()

        return AsymmetricStencil3D()
