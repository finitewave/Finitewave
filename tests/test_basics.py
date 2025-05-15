import os
import shutil
import numpy as np
import pytest
import finitewave as fw


def test_state_loading():
    n = 5
    tissue = fw.CardiacTissue2D([n, n])

    stim_sequence = fw.StimSequence()
    stim_sequence.add_stim(fw.StimCurrentCoord2D(0, 10, 0.5, 1, n//2, n//2 + 1,
                                                 n//2, n//2 + 1))

    state_saver = fw.StateSaverCollection()
    state_saver.savers.append(fw.StateSaver("state_0", time=3))

    model = fw.FentonKarma2D()
    model.dt = 0.01
    model.dr = 0.25
    model.t_max = 5

    model.cardiac_tissue = tissue
    model.stim_sequence = stim_sequence
    model.state_saver = state_saver

    model.run()

    u_before = model.u.copy()
    v_before = model.v.copy()
    w_before = model.w.copy()

    # recreate the model
    model = fw.FentonKarma2D()
    model.dt = 0.01
    model.dr = 0.25
    model.t_max = 5

    model.cardiac_tissue = tissue
    model.state_loader = fw.StateLoader("state_0")

    model.run()
    u_after = model.u.copy()
    v_after = model.v.copy()
    w_after = model.w.copy()

    assert np.allclose(u_before, u_after, atol=1e-5), "u states are not equal"
    assert np.allclose(v_before, v_after, atol=1e-5), "v states are not equal"
    assert np.allclose(w_before, w_after, atol=1e-5), "w states are not equal"

def test_commands():
    n = 5
    tissue = fw.CardiacTissue2D([n, n])

    stim_sequence = fw.StimSequence()

    model = fw.FentonKarma2D()
    model.dt = 0.01
    model.dr = 0.25
    model.t_max = 10

    class ExcitationCommand(fw.Command):
        def execute(self, model):
            model.u[1:-1, 1:-1] = 1

    command_sequence = fw.CommandSequence()
    command_sequence.add_command(ExcitationCommand(5))

    model.cardiac_tissue = tissue
    model.stim_sequence = stim_sequence
    model.command_sequence = command_sequence

    model.run()   

    assert np.mean(model.u[1:-1, 1:-1]) > 0.5, "Command did not work"