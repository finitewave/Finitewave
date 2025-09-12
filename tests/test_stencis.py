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
    w = np.array(w) * model.D_model * model.dt / model.dr ** 2
    w[2] += 1

    msg = "weights are not equal for conductivity"
    np.testing.assert_allclose(w, comp, atol=1e-5, err_msg=msg)

    mesh, weights = generate_2d_patterns()

    for m, w in zip(mesh, weights):
        tissue.mesh[1:-1, 1:-1] = np.array(m)
        tissue.conductivity = 1
        iso_stencil = fw.IsotropicStencil2D()
        comp = iso_stencil.compute_weights(model, tissue)
        comp = comp[2, 2, :]
        w = np.array(w) * model.D_model * model.dt / model.dr ** 2
        w[2] += 1
        msg = f"weights are not equal for {m}"
        np.testing.assert_allclose(w, comp, atol=1e-5, err_msg=msg)


def test_isotropic_stencil_3d():

    class DummyModel:
        def __init__(self):
            self.dt = 0.01
            self.dr = 0.25
            self.D_model = 1.
            self.npfloat = np.float64

    class DummyTissue:
        def __init__(self):
            self.mesh = np.zeros((5, 5, 5), dtype=np.int8)
            self.mesh[1:-1, 1:-1, 1:-1] = 1
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
    w = np.array(w) * model.D_model * model.dt / model.dr ** 2
    w[3] += 1

    msg = "weights are not equal for conductivity"
    np.testing.assert_allclose(w, comp, atol=1e-5, err_msg=msg)

    mesh, weights = generate_3d_patterns()

    for m, w in zip(mesh, weights):
        tissue.mesh[1:-1, 1:-1, 1:-1] = np.array(m)
        tissue.conductivity = 1
        iso_stencil = fw.IsotropicStencil3D()
        comp = iso_stencil.compute_weights(model, tissue)
        comp = comp[2, 2, 2, :]
        w = np.array(w) * model.D_model * model.dt / model.dr ** 2
        w[3] += 1
        msg = f"weights are not equal for {m}"
        np.testing.assert_allclose(w, comp, atol=1e-5, err_msg=msg)


def test_asymmetric_stencil_2d():
    class DummyModel:
        def __init__(self):
            self.dt = 1
            self.dr = 1
            self.D_model = 1.
            self.npfloat = np.float64

    class DummyTissue:
        def __init__(self, theta=np.pi/4):
            self.mesh = np.zeros((5, 5), dtype=np.int8)
            self.mesh[1:-1, 1:-1] = 1
            self.conductivity = np.ones((5, 5), dtype=np.float32)
            self.fibers = np.zeros((5, 5, 2), dtype=np.float32)
            self.fibers[:, :, 0] = np.cos(theta)
            self.fibers[:, :, 1] = np.sin(theta)

    D_al = 1.
    D_ac = 1 / 3

    for theta in [0., np.pi/6, np.pi/4]:
        model = DummyModel()
        tissue = DummyTissue(theta)

        d_xx = D_ac + (D_al - D_ac) * np.cos(theta) ** 2
        d_yy = D_ac + (D_al - D_ac) * np.sin(theta) ** 2
        d_xy = (D_al - D_ac) * np.cos(theta) * np.sin(theta)
        d_yx = (D_al - D_ac) * np.sin(theta) * np.cos(theta)

        mesh, weights = generate_assymmetric_2d_patterns(d_xx, d_yy, d_xy, d_yx)
        for m, w in zip(mesh, weights):
            tissue.mesh[1:-1, 1:-1] = np.array(m)
            tissue.conductivity = 1
            stencil = fw.AsymmetricStencil2D()
            stencil.D_al = D_al
            stencil.D_ac = D_ac
            comp = stencil.compute_weights(model, tissue)
            comp = comp[2, 2, :]
            w = np.array(w) * model.D_model * model.dt / model.dr ** 2
            w[4] += 1

            msg = f"weights are not equal for {m}"
            np.testing.assert_allclose(comp, w, atol=1e-5, err_msg=msg)

    for theta in [0., np.pi/6, np.pi/4]:
        m = np.ones((3, 3))
        conductivity = np.ones((3, 3))
        i, j = 1, 1
        conductivity[i-1, j] = 0.2
        conductivity[i, j-1] = 0.4
        conductivity[i, j+1] = 0.6
        conductivity[i+1, j] = 0.8

        d_xx = D_ac + (D_al - D_ac) * np.cos(theta) ** 2
        d_yy = D_ac + (D_al - D_ac) * np.sin(theta) ** 2
        d_xy = (D_al - D_ac) * np.cos(theta) * np.sin(theta)
        d_yx = (D_al - D_ac) * np.sin(theta) * np.cos(theta)

        d_xx = [d_xx * 0.6, d_xx * 0.9]
        d_yy = [d_yy * 0.7, d_yy * 0.8]
        d_xy = [d_xy * 0.6, d_xy * 0.9]
        d_yx = [d_yx * 0.7, d_yx * 0.8]

        m, w = generate_assymmetric_2d_conductivity(d_xx, d_yy, d_xy, d_yx)

        w = np.array(w) * model.D_model * model.dt / model.dr ** 2
        w[4] += 1

        model = DummyModel()
        tissue = DummyTissue(theta)
        tissue.mesh[1:-1, 1:-1] = np.array(m)
        tissue.conductivity = np.ones((5, 5))
        tissue.conductivity[1:-1, 1:-1] = conductivity
        stencil = fw.AsymmetricStencil2D()
        stencil.D_al = D_al
        stencil.D_ac = D_ac

        w_comp = stencil.compute_weights(model, tissue)
        w_comp = w_comp[2, 2, :]

        msg = "Test for heterogenius conductivity is failed"
        np.testing.assert_allclose(w_comp, w, atol=1e-5, err_msg=msg)


def generate_assymmetric_2d_conductivity(d_xx, d_yy, d_xy, d_yx):

    m_base = np.ones((3, 3))

    m = m_base.copy()
    w = np.array([
        1/4 * (d_xy[0] + d_yx[0]),                                           # i-1, j-1
        d_xx[0] - 1/4 * (d_xy[0] + d_yx[1]) + 1/4 * (d_xy[0] + d_yx[0]),     # i-1, j
        - 1/4 * (d_xy[0] + d_yx[1]),                                         # i-1, j+1
        d_yy[0] - 1/4 * (d_xy[1] + d_yx[0]) + 1/4 * (d_xy[0] + d_yx[0]),     # i, j-1
        0.,                                                   # i, j
        d_yy[1] - 1/4 * (d_xy[0] + d_yx[1]) + 1/4 * (d_xy[1] + d_yx[1]),     # i, j+1
        - 1/4 * (d_xy[1] + d_yx[0]),                                # i+1, j-1
        d_xx[1] - 1/4 * (d_xy[1] + d_yx[0]) + 1/4 * (d_xy[1] + d_yx[1]),     # i+1, j
        1/4 * (d_xy[1] + d_yx[1])                                   # i+1, j+1
    ])
    w[4] = -np.sum(w)
    return m, w


def generate_assymmetric_2d_patterns(d_xx, d_yy, d_xy, d_yx):
    mesh = []
    weights = []

    m_base = np.ones((3, 3))
    i, j = 1, 1

    m = m_base.copy()
    mesh.append(m)
    w = np.array([
        1/4 * (d_xy + d_yx),                                  # i-1, j-1
        d_xx - 1/4 * (d_xy + d_yx) + 1/4 * (d_xy + d_yx),     # i-1, j
        - 1/4 * (d_xy + d_yx),                                # i-1, j+1
        d_yy - 1/4 * (d_xy + d_yx) + 1/4 * (d_xy + d_yx),     # i, j-1
        0.,                                                   # i, j
        d_yy - 1/4 * (d_xy + d_yx) + 1/4 * (d_xy + d_yx),     # i, j+1
        - 1/4 * (d_xy + d_yx),                                # i+1, j-1
        d_xx - 1/4 * (d_xy + d_yx) + 1/4 * (d_xy + d_yx),     # i+1, j
        1/4 * (d_xy + d_yx)                                   # i+1, j+1
    ])
    w[4] = -np.sum(w)
    weights.append(w)

    m = m_base.copy()
    m[i-1, j-1] = 2
    mesh.append(m)
    w = np.array([
        0 * (d_xy + d_yx),                                     # i-1, j-1
        d_xx - 1/4 * (d_xy + d_yx) + 1/3 * (d_xy + d_yx),     # i-1, j
        - 1/4 * (d_xy + d_yx),                                # i-1, j+1
        d_yy - 1/4 * (d_xy + d_yx) + 1/3 * (d_xy + d_yx),     # i, j-1
        0.,                                                    # i, j
        d_yy - 1/4 * (d_xy + d_yx) + 1/4 * (d_xy + d_yx),    # i, j+1
        - 1/4 * (d_xy + d_yx),                                # i+1, j-1
        d_xx - 1/4 * (d_xy + d_yx) + 1/4 * (d_xy + d_yx),    # i+1, j
        1/4 * (d_xy + d_yx)                                   # i+1, j+1
    ])
    w[4] = -np.sum(w)
    weights.append(w)

    m = m_base.copy()
    m[i-1, j] = 2
    mesh.append(m)
    w = np.array([
        0 * d_xy + 1/3 * d_yx,                                              # i-1, j-1
        0 * d_xx - (0 * d_xy + 1/3 * d_yx) + (0 * d_xy + 1/3 * d_yx),       # i-1, j
        - (0 * d_xy + 1/3 * d_yx),                                          # i-1, j+1
        1 * d_yy - (1/4 * d_xy + 1/4 * d_yx) + (0 * d_xy + 1/3 * d_yx),     # i, j-1
        0.,                                                                 # i, j
        1 * d_yy - (0 * d_xy + 1/3 * d_yx) + (1/4 * d_xy + 1/4 * d_yx),     # i, j+1
        - (1/4 * d_xy + 1/4 * d_yx),                                        # i+1, j-1
        1 * d_xx - (1/4 * d_xy + 1/4 * d_yx) + (1/4 * d_xy + 1/4 * d_yx),   # i+1, j
        (1/4 * d_xy + 1/4 * d_yx)                                           # i+1, j+1
    ])
    w[4] = -np.sum(w)
    weights.append(w)

    m = m_base.copy()
    m[i-1, j+1] = 2
    mesh.append(m)
    w = np.array([
        1/4 * (d_xy + d_yx),                                     # i-1, j-1
        d_xx - 1/3 * (d_xy + d_yx) + 1/4 * (d_xy + d_yx),     # i-1, j
        - 0 * (d_xy + d_yx),                                # i-1, j+1
        d_yy - 1/4 * (d_xy + d_yx) + 1/4 * (d_xy + d_yx),     # i, j-1
        0.,                                                    # i, j
        d_yy - 1/3 * (d_xy + d_yx) + 1/4 * (d_xy + d_yx),    # i, j+1
        - 1/4 * (d_xy + d_yx),                                # i+1, j-1
        d_xx - 1/4 * (d_xy + d_yx) + 1/4 * (d_xy + d_yx),    # i+1, j
        1/4 * (d_xy + d_yx)                                   # i+1, j+1
    ])
    w[4] = -np.sum(w)
    weights.append(w)

    m = m_base.copy()
    m[i, j-1] = 2
    mesh.append(m)
    w = np.array([
        1/3 * d_xy + 0 * d_yx,                                              # i-1, j-1
        1. * d_xx - (1/4 * d_xy + 1/4 * d_yx) + (0 * d_xy + 1/3 * d_yx),    # i-1, j
        - (1/4 * d_xy + 1/4 * d_yx),                                          # i-1, j+1
        0 * d_yy - (1/3 * d_xy + 0 * d_yx) + (1/3 * d_xy + 0 * d_yx),     # i, j-1
        0.,                                                                 # i, j
        1 * d_yy - (1/4 * d_xy + 1/4 * d_yx) + (1/4 * d_xy + 1/4 * d_yx),     # i, j+1
        - (1/3 * d_xy + 0 * d_yx),                                        # i+1, j-1
        1 * d_xx - (1/3 * d_xy + 0 * d_yx) + (1/4 * d_xy + 1/4 * d_yx),   # i+1, j
        (1/4 * d_xy + 1/4 * d_yx)                                           # i+1, j+1
    ])
    w[4] = -np.sum(w)
    weights.append(w)

    # use symmetry

    m = m_base.copy()
    m[i, j+1] = 2
    mesh.append(m)
    w = weights[4].copy()[::-1]
    weights.append(w)

    m = m_base.copy()
    m[i+1, j-1] = 2
    mesh.append(m)
    w = weights[3].copy()[::-1]
    weights.append(w)

    m = m_base.copy()
    m[i+1, j] = 2
    mesh.append(m)
    w = weights[2].copy()[::-1]
    weights.append(w)

    m = m_base.copy()
    m[i+1, j+1] = 2
    mesh.append(m)
    w = weights[1].copy()[::-1]
    weights.append(w)

    return mesh, weights


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
