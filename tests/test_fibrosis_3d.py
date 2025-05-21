import random
import numpy as np
from finitewave.cpuwave3D.fibrosis.diffuse_3d_pattern import Diffuse3DPattern
from finitewave.cpuwave3D.fibrosis.structural_3d_pattern import Structural3DPattern

def test_diffuse_fibrosis_3d():
    shape = (100, 100, 100)
    x1, x2 = 10, 90
    y1, y2 = 20, 80
    z1, z2 = 30, 70
    density = 0.3

    random.seed(0)

    pattern = Diffuse3DPattern(density=density, x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2)
    result = pattern.generate(shape=shape)

    assert result.shape == shape, "Diffuse: shape mismatch"

    assert np.all(np.isin(result, [1, 2])), "Diffuse: invalid values in result"

    subregion = result[x1:x2, y1:y2, z1:z2]
    fibrosis_ratio = np.sum(subregion == 2) / subregion.size

    assert abs(fibrosis_ratio - density) < 0.01, "Diffuse: fibrosis density mismatch"

def test_structural_fibrosis_3d():
    shape = (100, 100, 100)
    x1, x2 = 10, 90
    y1, y2 = 20, 80
    z1, z2 = 30, 70
    density = 0.4
    length_i = 5
    length_j = 4
    length_k = 3

    random.seed(0)

    pattern = Structural3DPattern(
        density=density, length_i=length_i, length_j=length_j, length_k=length_k,
        x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2
    )
    result = pattern.generate(shape=shape)

    assert result.shape == shape, "Structural: shape mismatch"
    assert np.all(np.isin(result, [1, 2])), "Structural: invalid values in result"

    subregion = result[x1:x2, y1:y2, z1:z2]
    fibrosis_ratio = np.sum(subregion == 2) / subregion.size

    assert abs(fibrosis_ratio - density) < 0.05, "Structural: fibrosis density mismatch"