from finitewave.numerics.time_integration.implicit_time_integration import ImplicitTimeIntegration



class BackwardEulerTimeIntegration(ImplicitTimeIntegration):
    """Advance cardiac simulations with Backward Euler time integration.

    Diffusion is treated implicitly using the Backward Euler method. The
    reaction contribution is supplied explicitly by the cardiac model, and
    the resulting linear system is solved with Conjugate Gradient by default.

    Parameters
    ----------
    atol : float, optional
        Absolute tolerance for the linear solver. Default is 1e-8.
    maxiter : int, optional
        Maximum number of linear solver iterations per time step. Default is
        100.
    lumping_factor : float, optional
        Interpolation factor between the consistent and row-sum lumped mass
        matrices. Use 0 for the consistent matrix and 1 for full lumping.
        Default is 0.0.
    reaction_lumping : bool, optional
        If True, use the row-sum lumped mass matrix for the reaction term;
        otherwise, use the consistent mass matrix. Default is False.
        This setting is independent of ``lumping_factor``.

    Attributes
    ----------
    order : int
        Diffusion method order, fixed to 1.
    num_iterations : list of int
        Linear solver iteration counts recorded at each time step. A negative
        value indicates that the requested accuracy was not reached.

    Notes
    -----
    Each step solves ``(M_eff + dt * K) @ u_new = M_eff @ u_old + dt * M_r @ f``,
    where ``K`` is the stiffness matrix, ``M_eff`` is the mass matrix blended
    according to ``lumping_factor``, ``M_r`` is the reaction mass matrix, and
    ``f`` is the reaction term supplied by the cardiac model.

    Examples
    --------
    Run a short Aliev-Panfilov simulation on a 2D grid with a voltage stimulus.
    The simulation initializes the integrator and advances it at each time step.

    >>> import finitewave as fw
    ...
    >>> sim = fw.CardiacSimulation(dt=0.01, t_max=1.0, backend="numba")
    >>> sim.cardiac_tissue = fw.CardiacTissue(shape=(20, 20), dr=0.25)
    >>> sim.cardiac_model = fw.AlievPanfilov()
    >>> sim.time_integration = fw.BackwardEulerTimeIntegration(atol=1e-6,
    ...                                                        maxiter=100,
    ...                                                        lumping_factor=0.0,
    ...                                                        reaction_lumping=True)
    >>> sim.stim_sequence = fw.StimSequence()
    >>> _ = sim.stim_sequence.add_stim(
    ...         fw.StimVoltageCoord(time=0.0, volt_value=1.0,
    ...                             x_min=0, x_max=5, y_min=0, y_max=20)
    ... )
    >>> sim.run(prog_bar=False)
    >>> u = sim.cardiac_model.u
    """

    def __init__(self, atol=1e-8, maxiter=100, lumping_factor=0.0,
                 reaction_lumping=False):
        super().__init__(atol=atol, maxiter=maxiter, lumping_factor=lumping_factor,
                         order=1, reaction_lumping=reaction_lumping)
