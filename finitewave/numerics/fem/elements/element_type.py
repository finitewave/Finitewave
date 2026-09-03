from .quadrilateral_element import LinearQuadrilateralElement
from .triangle_element import LinearTriangleElement
from .tetrahedral_element import LinearTetrahedralElement


class ElementType:
    """
    Enumeration of element types for finite element meshes.
    """
    TRIANGLE = "Triangle"
    QUAD = "Quadrilateral"
    TETRA = "Tetrahedra"

    values = [TRIANGLE, QUAD, TETRA]
    surface = [TRIANGLE, QUAD]
    volume = [TETRA]

    @staticmethod
    def is_valid(name):
        return name in ElementType.values

    @staticmethod
    def select_reference_element(name, order=1):
        """
        Selects the appropriate reference element class based on the element type.

        Parameters
        ----------
        name : str
            The name of the element type.
        order : int
            The order of the element (default is 1).

        Returns
        -------
        ReferenceElement
            An instance of the corresponding reference element class.

        Raises
        ------
        ValueError
            If the provided name is not a valid element type.
        """
        if name == ElementType.TRIANGLE and order == 1:
            return LinearTriangleElement()
        elif name == ElementType.QUAD and order == 1:
            return LinearQuadrilateralElement()
        elif name == ElementType.TETRA and order == 1:
            return LinearTetrahedralElement()
        else:
            raise ValueError(f"Invalid element type: {name}, or order: {order}.")
