from .implicit_time_integration import ImplicitTimeIntegration



class BackwardEulerTimeIntegration(ImplicitTimeIntegration):
    def __init__(self, atol=1e-8, maxiter=100, lumping_factor=0.0,
                 reaction_lumping=False):
        super().__init__(atol=atol, maxiter=maxiter, lumping_factor=lumping_factor,
                         order=1, reaction_lumping=reaction_lumping)
