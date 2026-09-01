

class Preconditioner:
    def __init__(self, solver):
        self.solver = solver

    def build(self, x):
        raise NotImplementedError("Preconditioner must implement the build method.")