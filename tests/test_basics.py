import numpy as np

import finitewave as fw
from finitewave.core.command import Command, CommandSequence
from finitewave.core.state import StateLoader, StateSaver


class StateModel:
    D_model = 0.0
    state_vars = ("u", "v")

    def initialize(self, simulation):
        n_points = simulation.cardiac_tissue.tissue_indexes.size
        self._u = np.zeros(n_points)
        self._v = np.ones(n_points)
        self._rhs = np.zeros(n_points)

    @property
    def u(self):
        return self._u.copy()

    @property
    def v(self):
        return self._v.copy()

    def update_state_variables(self, values):
        for name, value in values.items():
            setattr(self, f"_{name}", np.array(value, copy=True))

    def run(self):
        self._v += 0.1

    def sync_backend(self):
        pass


def _state_simulation(t_max):
    simulation = fw.CardiacSimulation(dt=0.01, t_max=t_max)
    simulation.cardiac_tissue = fw.CardiacTissue(
        shape=(5, 5), dr=0.25
    )
    simulation.cardiac_model = StateModel()
    return simulation


def test_state_loading(tmp_path):
    state_path = tmp_path / "state"
    simulation = _state_simulation(t_max=0.05)
    simulation.state_saver = StateSaver(
        str(state_path), time=0.03
    )

    simulation.run(prog_bar=False)
    expected = {
        name: getattr(simulation.cardiac_model, name).copy()
        for name in simulation.cardiac_model.state_vars
    }

    resumed = _state_simulation(t_max=0.02)
    resumed.state_loader = StateLoader(str(state_path))
    resumed.run(prog_bar=False)

    for name, expected_values in expected.items():
        np.testing.assert_allclose(
            getattr(resumed.cardiac_model, name),
            expected_values,
            atol=1e-12,
        )


def test_commands_receive_simulation():
    simulation = _state_simulation(t_max=0.03)

    class RecordExecution(Command):
        def execute(self, active_simulation):
            active_simulation.meta["command_executed"] = True

    simulation.command_sequence = CommandSequence()
    simulation.command_sequence.add_command(RecordExecution(time=0.01))

    simulation.run(prog_bar=False)

    assert simulation.meta["command_executed"] is True
