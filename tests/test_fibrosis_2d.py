import random
import numpy as np
from finitewave.cpuwave2D.fibrosis.diffuse_2d_pattern import Diffuse2DPattern
from finitewave.cpuwave2D.fibrosis.structural_2d_pattern import Structural2DPattern

def test_diffuse_fibrosis_2d():
    shape = (100, 100)
    x1, x2 = 10, 90
    y1, y2 = 20, 80
    density = 0.3

    random.seed(0)

    pattern = Diffuse2DPattern(density=density, x1=x1, x2=x2, y1=y1, y2=y2)
    result = pattern.generate(shape=shape)

    assert result.shape == shape, "Diffuse: shape mismatch"

    assert np.all(np.isin(result, [1, 2])), "Diffuse: invalid values in result"

    subregion = result[x1:x2, y1:y2]
    fibrosis_ratio = np.sum(subregion == 2) / subregion.size

    assert abs(fibrosis_ratio - density) < 0.01, "Diffuse: fibrosis density mismatch"

def test_structural_fibrosis_2d():
    shape = (100, 100)
    x1, x2 = 10, 90
    y1, y2 = 20, 80
    density = 0.4
    length_i = 5
    length_j = 4

    random.seed(0)

    pattern = Structural2DPattern(
        density=density, length_i=length_i, length_j=length_j,
        x1=x1, x2=x2, y1=y1, y2=y2
    )
    result = pattern.generate(shape=shape)

    assert result.shape == shape, "Structural: shape mismatch"
    assert np.all(np.isin(result, [1, 2])), "Structural: invalid values in result"

    subregion = result[x1:x2, y1:y2]
    fibrosis_ratio = np.sum(subregion == 2) / subregion.size

    assert abs(fibrosis_ratio - density) < 0.05, "Structural: fibrosis density mismatch"