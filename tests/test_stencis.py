import numpy as np

import finitewave as fw

def generate_patterns():
    mesh = []
    w = []

    mesh.append([[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]])
    
    w.append([1, 1, -4, 1, 1])

    mesh.append([[1, 2, 1],
            [1, 1, 1],
            [1, 1, 1]])
    
    w.append([1, 2, -4, 0, 1])
    
    mesh.append([[1, 1, 1],
            [1, 1, 2],
            [1, 1, 1]])
    
    w.append([2, 1, -4, 1, 0])
    
    mesh.append([[1, 1, 1],
            [1, 1, 1],
            [1, 2, 1]])
    
    w.append([1, 0, -4, 2, 1])
    
    mesh.append([[1, 1, 1],
            [2, 1, 1],
            [1, 1, 1]])
    
    w.append([0, 1, -4, 1, 2])
    
    mesh.append([[1, 2, 1],
            [1, 1, 1],
            [1, 2, 1]])
    
    w.append([1, 0, -2, 0, 1])
    
    mesh.append([[1, 1, 1],
            [2, 1, 2],
            [1, 1, 1]])
    
    w.append([0, 1, -2, 1, 0])
    
    mesh.append([[1, 2, 1],
            [2, 1, 2],
            [1, 2, 1]])
    
    w.append([0, 0, 0, 0, 0])
    return mesh, w


def test_iso_weights_2d():

    class DummyModel:
        def __init__(self):
            self.dt = 0.01
            self.dr = 0.25
            self.D_model = 1.
            self.npfloat = np.float64

    class DummyTissue:
        def __init__(self):
            self.mesh = np.array([[0, 0, 0, 0, 0],
                                  [0, 1, 1, 1, 0],
                                  [0, 1, 1, 1, 0],
                                  [0, 1, 1, 1, 0],
                                  [0, 0, 0, 0, 0]], dtype=np.int8)
            self.conductivity = 1.0
    
    model = DummyModel()
    tissue = DummyTissue()

    mesh, weights = generate_patterns()

    for m, w in zip(mesh, weights):
        tissue.mesh[1:-1, 1:-1] = np.array(m).T
        tissue.conductivity = 1
        iso_stencil = fw.IsotropicStencil2D()
        comp = iso_stencil.compute_weights(model, tissue)
        comp = comp[2, 2, :]
        w = np.array(w)  * model.D_model * model.dt / model.dr ** 2
        w[2] += 1
        assert np.allclose(w, comp, atol=10-5), f"weights are not equal for {m}"
