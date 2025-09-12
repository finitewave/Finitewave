from finitewave.core.tissue.cardiac_tissue import CardiacTissue


class CardiacTissueElements(CardiacTissue):
    def __init__(self, coords, elems):
        super().__init__()
        self.coords = coords
        self.elems = elems
        self.conductivity = 1.0
        self.fibers = None

    def add_boundaries(self):
        pass
