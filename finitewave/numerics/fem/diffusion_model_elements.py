import numpy as np
from finitewave.core.diffusion.diffusion_model_base import DiffusionModelBase
from finitewave.numerics.fem.element_assembler import ElementAssembler
from finitewave.numerics.fem.linear.linear_triangle_element import LinearTriangleElement
from finitewave.numerics.fem.linear.linear_quadrilateral_element import LinearQuadrilateralElement
from finitewave.numerics.fem.linear.linear_tetrahedral_element import LinearTetrahedralElement
from finitewave.core.numerics.fem.elements.element_type import ElementShape, ElementOrder

REFERENCE_ELEMENTS = {
    (ElementShape.TRIANGLE, ElementOrder.LINEAR): LinearTriangleElement,
    (ElementShape.QUAD, ElementOrder.LINEAR): LinearQuadrilateralElement,
    (ElementShape.TETRA, ElementOrder.LINEAR): LinearTetrahedralElement,
}

class DiffusionModelElements(DiffusionModelBase):
    """
    Class for assembling element-based diffusion operator.

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
        if self.reference_element is None:
            self.reference_element = self.default_element()
        self.update_weights()

    def update_weights(self):
        """
        Updates the weights (stiffness and mass matrices) for the
        element-based model. This can be used to recompute the weights if
        the tissue properties or diffusion tensor change during the simulation.
        """
        tissue = self.simulation.cardiac_tissue
        model = self.simulation.cardiac_model
        self.compute_weights(tissue, model.D_model)

    def compute_weights(self, tissue, D_model=1.0):
        """
        Computes the stiffness and mass matrices for the element-based model.

        Parameters
        ----------
        tissue : CardiacTissue
            The cardiac tissue for which to compute the diffusion weights.
        D_model : float, optional
            Model-specific diffusion coefficient. Default is 1.0.

        Returns
        -------
        scipy.sparse.csr_matrix
            The stiffness matrix for the element-based model.
        scipy.sparse.csr_matrix
            The mass matrix for the element-based model.
        """
        indexes = tissue.tissue_indexes[tissue.myo_indexes]
        coords = tissue.coords
        elems = tissue.myo_elements

        diffusion = self.compute_diffusion_tensor(tissue, D_model)
        diffusion = diffusion[tissue.myo_elems_indexes]

        self.weights = self.assembler.compute_system_matrices(
            coords, elems, diffusion, indexes, reindex=True
        )
        return self.weights

    def compute_diffusion_tensor(self, tissue, D_model=1.0):
        """
        Computes the diffusion tensor for each element.

        Parameters
        ----------
        tissue : CardiacTissue
            The cardiac tissue instance.
        D_model : float, optional
            Model-specific diffusion coefficient. Default is 1.0.

        Returns
        -------
        np.ndarray
            The diffusion tensor for each element.
        """
        d_ac = tissue.D_ac
        d_al = tissue.D_al

        dim_tissue = tissue.coords.shape[1]

        diffusion = np.eye(dim_tissue, dtype=np.float64)

        if tissue.fibers is not None:
            diffusion = (d_ac * np.eye(dim_tissue)[np.newaxis, :, :] +
                         ((d_al - d_ac) *
                          tissue.fibers[:, :, np.newaxis] @
                          tissue.fibers[:, np.newaxis, :]))

        conductivity = (D_model * tissue.conductivity * 
                        np.ones(len(tissue.elems), dtype=np.float64))
        diffusion = diffusion * conductivity[:, np.newaxis, np.newaxis]
        return diffusion
    
    def default_element(self):
        tissue = self.simulation.cardiac_tissue

        key = (tissue.element_shape, tissue.element_order)

        try:
            return REFERENCE_ELEMENTS[key]()
        except KeyError as e:
            supported = [(s.value, o.value) for s, o in REFERENCE_ELEMENTS]
            raise ValueError(
                f"Unsupported reference element: "
                f"shape={key[0].value}, order={key[1].value}. "
                f"Supported elements: {supported}."
            ) from e

