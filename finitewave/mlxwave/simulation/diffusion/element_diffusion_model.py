import numpy as np
from finitewave.core.diffusion.diffusion_model_base import DiffusionModelBase
from finitewave.cpuwave.numerics.fem.element_assembler import ElementAssembler


class ElementDiffusionModel(DiffusionModelBase):
    """
    Class for assembling element-based diffusion models.

    Attributes
    ----------
    assembler : ElementAssembler
        The assembler used to compute the stiffness and mass matrices for
        the element-based model.
    reference_element : ReferenceElement
        The reference element used for numerical integration.
    simulation : Simulation
        The simulation instance associated with this assembler.
    weights : tuple
        The computed diffusion weights (stiffness and mass matrices).
    """
    def __init__(self):
        super().__init__()
        self.assembler = ElementAssembler()
        self.simulation = None

    @property
    def reference_element(self):
        return self.assembler.reference_element
    
    @reference_element.setter
    def reference_element(self, value):
        self.assembler.reference_element = value

    def initialize(self, simulation):
        """
        Computes the weights (stiffness and mass matrices) for the
        element-based model.

        Parameters
        ----------
        simulation : Simulation
            The simulation instance associated with this assembler.
        """
        self.simulation = simulation
        self.compute_weights()

    def compute_weights(self):
        """
        Computes the stiffness and mass matrices for the element-based model.

        Returns
        -------
        scipy.sparse.csr_matrix
            The stiffness matrix for the element-based model.
        scipy.sparse.csr_matrix
            The mass matrix for the element-based model.
        """
        tissue = self.simulation.cardiac_tissue

        indexes = tissue.myo_indexes
        coords = tissue.coords
        elems = tissue.myo_elements

        diffusion = self.compute_diffusion(self.simulation, tissue)
        diffusion = diffusion[tissue.myo_elems_indexes]

        self.weights = self.assembler.compute_system_matrices(coords, elems,
                                                              diffusion,
                                                              indexes,
                                                              reindex=True)
        return self.weights

    def compute_diffusion(self, simulation, tissue):
        """
        Computes the diffusion tensor for each element.

        Parameters
        ----------
        simulation : Simulation
            The simulation instance.
        tissue : CardiacTissue
            The cardiac tissue instance.

        Returns
        -------
        np.ndarray
            The diffusion tensor for each element.
        """
        d_ac = tissue.D_ac
        d_al = tissue.D_al
        d_model = simulation.cardiac_model.D_model

        dim_tissue = tissue.coords.shape[1]

        diffusion = np.eye(dim_tissue, dtype=simulation.npfloat)

        if tissue.fibers is not None:
            diffusion = (d_ac * np.eye(dim_tissue)[np.newaxis, :, :] +
                         ((d_al - d_ac) *
                          tissue.fibers[:, :, np.newaxis] @
                          tissue.fibers[:, np.newaxis, :]))

        conductivity = (d_model * tissue.conductivity *
                        np.ones(len(tissue.elems), dtype=simulation.npfloat))
        diffusion = diffusion * conductivity[:, np.newaxis, np.newaxis]
        return diffusion
