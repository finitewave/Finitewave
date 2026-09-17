import copy

import numpy as np
import pytest

import finitewave as fw
from finitewave.core.stimul.stim_type.stim_voltage import StimVoltage


def test_public_arrays_and_numba_simulation():
    backend_name = "numba"
    model = fw.AlievPanfilov()
    assert model._u is None
    assert model.u is None
    parameter = model.state_pars[0]
    model.set_parameters({parameter: np.full(16, getattr(model, parameter))})
    simulation = fw.CardiacSimulation(dt=0.01, t_max=0.02, backend=backend_name)
    tissue = fw.CardiacTissueGrid((4, 4), dr=0.25)
    simulation.cardiac_tissue = tissue
    simulation.cardiac_model = model
    simulation.initialize()
    assert isinstance(getattr(model, parameter), np.ndarray)
    np.testing.assert_array_equal(
        getattr(model, parameter).ravel(), np.asarray(getattr(model, f"_{parameter}"))
    )
    stimulus = StimVoltage(time=0, volt_value=0.75)
    stimulus.stim_indexes = simulation.backend.wrap_indexes(np.array([0, 1]))
    stimulus.stimulate(simulation)
    np.testing.assert_allclose(model.u.ravel()[:2], 0.75)

    for name in model.state_vars:
        public = getattr(model, name)
        wrapped = getattr(model, f"_{name}")
        assert isinstance(public, np.ndarray)
        np.testing.assert_array_equal(public.ravel(), np.asarray(wrapped))
    assert model.u.shape == (4, 4)
    assert model.output("u").shape == (16,)

    # Explicit updates must reach the cached kernel inputs.
    other = next(name for name in model.state_vars if name != "u")
    setattr(model, other, np.full(16, 0.1))
    assert model.model_kernel_args[model.kernel_arg_names.index(other)] is getattr(model, f"_{other}")
    model.u = np.full((4, 4), 0.5)
    setattr(model, other, np.full((4, 4), 0.2))
    np.testing.assert_allclose(model.u, 0.5)
    np.testing.assert_allclose(getattr(model, other), 0.2)

    simulation.run(initialize=False, prog_bar=False)
    assert np.all(np.isfinite(model.u))
    assert not np.allclose(model.u, 0.5)
    for name in model.state_vars:
        np.testing.assert_array_equal(getattr(model, name).ravel(), np.asarray(getattr(model, f"_{name}")))
    cloned = copy.copy(model)
    np.testing.assert_array_equal(cloned.u, model.u)
    with pytest.raises(AttributeError):
        model.missing_variable


def test_public_array_expands_compact_model_storage():
    model = fw.AlievPanfilov()
    simulation = fw.CardiacSimulation(dt=0.01, t_max=0.02)
    simulation.cardiac_tissue = fw.CardiacTissueGrid((4, 4), dr=0.25)
    simulation.cardiac_tissue.mesh[0, 0] = 0
    model.initialize(simulation)
    assert model.u.shape == (4, 4)
    assert model.output("u").shape == (15,)
    assert np.isnan(model.u[0, 0])


def test_model_values_are_updated_through_assignment():
    backend_name = "numba"
    model = fw.AlievPanfilov()
    with pytest.raises(AttributeError, match="before initialization"):
        model.u = 1.
    model.init_u = 0.25
    simulation = fw.CardiacSimulation(dt=0.01, t_max=0.02, backend=backend_name)
    simulation.cardiac_tissue = fw.CardiacTissueGrid((4, 4), dr=0.25)
    simulation.cardiac_model = model
    simulation.initialize()
    parameter = model.state_pars[0]
    field = np.full((4, 4), getattr(model, parameter))
    model.set_parameters({parameter: field})
    field[:] = -123
    assert not np.any(getattr(model, parameter) == -123)
    values = np.full(16, 0.5)
    model.u = values
    np.testing.assert_allclose(model.u, 0.5)
    with pytest.raises(ValueError):
        model.u = np.zeros(3)
    np.testing.assert_allclose(model.u, 0.5)
    simulation.run(initialize=False, prog_bar=False)
    assert np.all(np.isfinite(model.u))
