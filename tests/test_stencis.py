import numpy as np

import finitewave as fw


def test_isotropic_stencil_2d():

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

    conductivity = np.ones((5, 5))
    conductivity[1:-1, 1:-1] = np.array([[0.4, 0.2, 1.0],
                                         [0.4, 1.0, 1.0],
                                         [0.4, 1.0, 1.0]])
    
    w = [0.6, 0.7, -3.3, 1.0, 1.0]

    tissue.conductivity = conductivity
    
    iso_stencil = fw.IsotropicStencil2D()
    comp = iso_stencil.compute_weights(model, tissue)
    comp = comp[2, 2, :]
    w = np.array(w)  * model.D_model * model.dt / model.dr ** 2
    w[2] += 1
    
    msg = "weights are not equal for conductivity"
    assert np.allclose(w, comp, atol=1e-5), msg

    
    mesh, weights = generate_2d_patterns()

    for m, w in zip(mesh, weights):
        tissue.mesh[1:-1, 1:-1] = np.array(m)
        tissue.conductivity = 1
        iso_stencil = fw.IsotropicStencil2D()
        comp = iso_stencil.compute_weights(model, tissue)
        comp = comp[2, 2, :]
        w = np.array(w)  * model.D_model * model.dt / model.dr ** 2
        w[2] += 1
        msg = f"weights are not equal for {m}"
        assert np.allclose(w, comp, atol=1e-5), msg


def test_isotropic_stencil_3d():

    class DummyModel:
        def __init__(self):
            self.dt = 0.01
            self.dr = 0.25
            self.D_model = 1.
            self.npfloat = np.float64

    class DummyTissue:
        def __init__(self):
            self.mesh = np.ones((5, 5, 5), dtype=np.int8)
            self.conductivity = np.ones((5, 5, 5), dtype=np.float32)
    
    model = DummyModel()
    tissue = DummyTissue()

    conductivity = np.ones((3, 3, 3))
    i, j, k = 1, 1, 1
    conductivity[i-1, j, k] = 0.0
    conductivity[i, j-1, k] = 0.2
    conductivity[i, j, k-1] = 0.4
    conductivity[i, j, k+1] = 0.6
    conductivity[i, j+1, k] = 0.8
    conductivity[i+1, j, k] = 1.0

    w = [0.5, 0.6, 0.7, -4.5, 0.8, 0.9, 1.0]

    tissue.conductivity[1:-1, 1:-1, 1:-1] = conductivity
    
    iso_stencil = fw.IsotropicStencil3D()
    comp = iso_stencil.compute_weights(model, tissue)
    comp = comp[2, 2, 2, :]
    w = np.array(w)  * model.D_model * model.dt / model.dr ** 2
    w[3] += 1
    
    msg = "weights are not equal for conductivity"
    assert np.allclose(w, comp, atol=1e-5), msg

    
    mesh, weights = generate_3d_patterns()

    for m, w in zip(mesh, weights):
        tissue.mesh[1:-1, 1:-1, 1:-1] = np.array(m)
        tissue.conductivity = 1
        iso_stencil = fw.IsotropicStencil3D()
        comp = iso_stencil.compute_weights(model, tissue)
        comp = comp[2, 2, 2, :]
        w = np.array(w)  * model.D_model * model.dt / model.dr ** 2
        w[3] += 1
        msg = f"weights are not equal for {m}"
        assert np.allclose(w, comp, atol=1e-5), msg


def generate_2d_patterns():
    mesh = []
    w = []

    m_base = np.ones((3, 3))
    i, j = 1, 1
    
    m = m_base.copy()
    mesh.append(m)
    w.append([1, 1, -4, 1, 1])

    m = m_base.copy()
    m[i-1, j] = 2
    w.append([0, 1, -4, 1, 2])
    
    m = m_base.copy()
    m[i, j-1] = 2
    w.append([1, 0, -4, 2, 1])

    m = m_base.copy()
    m[i, j] = 2
    w.append([0, 0, 0, 0, 0])

    m = m_base.copy()
    m[i, j+1] = 2
    w.append([1, 2, -4, 0, 1])

    m = m_base.copy()
    m[i+1, j] = 2
    w.append([2, 1, -4, 1, 0])
    
    m = m_base.copy()
    m[i-1, j] = 2
    m[i+1, j] = 2
    w.append([0, 1, -2, 1, 0])

    m = m_base.copy()
    m[i, j-1] = 2
    m[i, j+1] = 2
    w.append([1, 0, -2, 0, 1])
    
    return mesh, w


def generate_3d_patterns():
    mesh = []
    w = []
    i, j, k = 1, 1, 1
    m_base = np.ones((3, 3, 3))

    m = m_base.copy()
    mesh.append(m)
    w.append([1, 1, 1, -6, 1, 1, 1])

    m = m_base.copy()
    m[i-1, j, k] = 2
    mesh.append(m)
    w.append([0, 1, 1, -6, 1, 1, 2])

    m = m_base.copy()
    m[i, j-1, k] = 2
    mesh.append(m)
    w.append([1, 0, 1, -6, 1, 2, 1])

    m = m_base.copy()
    m[i, j, k-1] = 2
    mesh.append(m)
    w.append([1, 1, 0, -6, 2, 1, 1])

    m = m_base.copy()
    m[i, j, k] = 2
    mesh.append(m)
    w.append([0, 0, 0, 0, 0, 0, 0])

    m = m_base.copy()
    m[i, j, k+1] = 2
    mesh.append(m)
    w.append([1, 1, 2, -6, 0, 1, 1])

    m = m_base.copy()
    m[i, j+1, k] = 2
    mesh.append(m)
    w.append([1, 2, 1, -6, 1, 0, 1])

    m = m_base.copy()
    m[i+1, j, k] = 2
    mesh.append(m)
    w.append([2, 1, 1, -6, 1, 1, 0])

    m = m_base.copy()
    m[i-1, j, k] = 2
    m[i+1, j, k] = 2
    mesh.append(m)
    w.append([0, 1, 1, -4, 1, 1, 0])

    m = m_base.copy()
    m[i, j-1, k] = 2
    m[i, j+1, k] = 2
    mesh.append(m)
    w.append([1, 0, 1, -4, 1, 0, 1])

    m = m_base.copy()
    m[i, j, k-1] = 2
    m[i, j, k+1] = 2
    mesh.append(m)
    w.append([1, 1, 0, -4, 0, 1, 1])

    return mesh, w
