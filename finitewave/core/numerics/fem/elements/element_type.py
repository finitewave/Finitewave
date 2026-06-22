from enum import Enum


class ElementShape(str, Enum):
    TRIANGLE = "Triangle"
    QUAD = "Quadrilateral"
    TETRA = "Tetrahedron"

class ElementOrder(str, Enum):
    LINEAR = "Linear"
    # QUADRATIC = "Quadratic"


SURFACE_ELEMENTS = {
    ElementShape.TRIANGLE, 
    ElementShape.QUAD
}
VOLUME_ELEMENTS = {
    ElementShape.TETRA
}
