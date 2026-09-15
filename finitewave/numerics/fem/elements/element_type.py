class ElementType:
    """
    Enumeration of element types for finite element meshes.
    """
    TRIANGLE = "Triangle"
    QUAD = "Quadrilateral"
    TETRA = "Tetrahedral"
    HEXAHEDRON = "Hexahedron"

    values = [TRIANGLE, QUAD, TETRA, HEXAHEDRON]
    surface = [TRIANGLE, QUAD]
    volume = [TETRA, HEXAHEDRON]

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
        from .hexahedral_element import LinearHexahedralElement
        from .quadrilateral_element import LinearQuadrilateralElement
        from .tetrahedral_element import LinearTetrahedralElement
        from .triangle_element import LinearTriangleElement

        element_classes = {
            ElementType.TRIANGLE: LinearTriangleElement,
            ElementType.QUAD: LinearQuadrilateralElement,
            ElementType.TETRA: LinearTetrahedralElement,
            ElementType.HEXAHEDRON: LinearHexahedralElement,
        }

        if name not in element_classes or order != 1:
            raise ValueError(f"Invalid element type: {name}, or order: {order}.")

        return element_classes[name]()
