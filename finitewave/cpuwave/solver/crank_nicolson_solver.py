from .implicit_solver import ImplicitSolver


class CrankNicolsonSolver(ImplicitSolver):
    def __init__(self, atol=1e-8, maxiter=100, lumping_factor=0.0, full_lumping=True):
        super().__init__(atol=atol, maxiter=maxiter, lumping_factor=lumping_factor, 
                         order=2, full_lumping=full_lumping)