import numpy as np
from finitewave.cpuwave.numerics.fem.surface_assembler import SurfaceAssembler
from .volume_model import VolumeModel


class SurfaceModel(VolumeModel):
    """
    Class for assembling element-based diffusion models.

    Attributes
    ----------
    reference_element : ReferenceElement
        The reference element used for numerical integration.
    simulation : Simulation
        The simulation instance associated with this assembler.
    weights : tuple
        The computed diffusion weights (stiffness and mass matrices).
    """
    def __init__(self):
        super().__init__()
        self.assembler = SurfaceAssembler()
