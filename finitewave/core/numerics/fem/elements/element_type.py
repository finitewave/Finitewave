

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
