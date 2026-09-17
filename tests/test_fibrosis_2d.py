import random

import numpy as np
import pytest

from finitewave.simulation.fibrosis import Diffuse2DPattern, Structural2DPattern


def test_diffuse_fibrosis_2d():
    shape = (1000, 1000)
    x1, x2 = 100, 900
    y1, y2 = 200, 800
    density = 0.3
    np.random.seed(0)

    pattern = Diffuse2DPattern(
        density=density, x1=x1, x2=x2, y1=y1, y2=y2
    )
    result = pattern.generate(shape=shape)

    assert result.shape == shape
    assert np.all(np.isin(result, [1, 2]))
    fibrosis_ratio = np.mean(result[x1:x2, y1:y2] == 2)
    assert fibrosis_ratio == pytest.approx(density, abs=0.01)


def test_structural_fibrosis_2d():
    shape = (1000, 1000)
    x1, x2 = 100, 900
    y1, y2 = 200, 800
    density = 0.4
    random.seed(0)

    pattern = Structural2DPattern(
        density=density,
        length_i=5,
        length_j=4,
        x1=x1,
        x2=x2,
        y1=y1,
        y2=y2,
    )
    result = pattern.generate(shape=shape)

    assert result.shape == shape
    assert np.all(np.isin(result, [1, 2]))
    fibrosis_ratio = np.mean(result[x1:x2, y1:y2] == 2)
    assert fibrosis_ratio == pytest.approx(density, abs=0.05)
