from abc import ABC, abstractmethod


def select_explicit_solver(x, active_indexes):
    raise NotImplementedError("select_explicit_solver must be implemented.")
    

def explicit_step(A_x, x, A_y, y, active_indexes, out):
    raise NotImplementedError("explicit_step must be implemented.")


def cg(A, b, x0=None, tol=1e-8, maxiter=None):
    raise NotImplementedError("cg must be implemented.")


def prepare_implicit_step(A_rhs, A_ion, x_old, x_old_2, i_ion, active_indexes):
    raise NotImplementedError("prepare_implicit_step must be implemented.")


def update_at_active_indexes(x, active_indexes):
    raise NotImplementedError("update_at_active_indexes must be implemented.")

