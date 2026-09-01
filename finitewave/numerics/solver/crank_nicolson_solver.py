from .implicit_solver import ImplicitSolver


class CrankNicolsonSolver(ImplicitSolver):
    def __init__(self, atol=1e-8, maxiter=100, lumping_factor=0.0, ionic_lumping=False):
        super().__init__(atol=atol, maxiter=maxiter, lumping_factor=lumping_factor, 
                         order=2, ionic_lumping=ionic_lumping)