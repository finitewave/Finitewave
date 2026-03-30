import numpy as np
import finitewave as fw
import matplotlib.pyplot as plt


def test_isotropic_stencil(mesh, diffusion, dr, indexes):
    stencil = fw.IsotropicStencil()
    K, M = stencil.compute_system_matrices(mesh, diffusion, dr, indexes)
    plt.imshow(K.toarray(), cmap='viridis')
    plt.colorbar()
    plt.show()


def test_asymmetric_stencil(mesh, diffusion, dr, indexes):
    stencil = fw.AsymmetricStencil()
    K, M = stencil.compute_system_matrices(mesh, diffusion, dr, indexes)
    plt.imshow(K.toarray(), cmap='viridis', origin='lower')
    plt.colorbar()
    plt.show()


def test_symmetric_stencil(mesh, diffusion, dr, indexes):
    cond = mesh.copy()
    cond[-1, :] = 0
    cond[:, -1] = 0
    # cond[2, 2] = 0
    # cond[2, 1] = 0
    # cond[1, 2] = 0
    # cond[1, 1] = 0
    indexes = np.flatnonzero(cond == 1)
    print(indexes)

    stencil = fw.SymmetricStencil()
    K, M = stencil.compute_system_matrices(mesh, diffusion, dr, indexes)

    K = np.ma.masked_array(K.toarray(), mask=(K.toarray() == 0))
    plt.imshow(K, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.show()

def test_element_assembler(coords, elems):
    assembler = fw.ElementAssembler()
    assembler.reference_element = fw.LinearTriangleElement()
    
    elements_size, grads = assembler.compute_metrics(coords, elems)
    print("Elements size:", elements_size)
    print("Gradients shape:", grads.shape)


alpha = np.pi / 4
D_al = 3.0
D_ac = 1.0
mesh = np.ones((3, 3))
mesh[2, 2] = 0
diffusion = np.zeros(mesh.shape + (mesh.ndim, mesh.ndim))
diffusion[..., 0, 0] = D_ac + (D_al - D_ac) * np.cos(alpha) * np.cos(alpha)
diffusion[..., 0, 1] = (D_al - D_ac) * np.cos(alpha) * np.sin(alpha)
diffusion[..., 1, 0] = (D_al - D_ac) * np.sin(alpha) * np.cos(alpha)
diffusion[..., 1, 1] = D_ac + (D_al - D_ac) * np.sin(alpha) * np.sin(alpha)
dr = 1.0
indexes = np.flatnonzero(mesh == 1)

# test_isotropic_stencil(mesh, diffusion, dr, indexes)
# test_asymmetric_stencil(mesh, diffusion, dr, indexes)
# test_symmetric_stencil(mesh, diffusion, dr, indexes)

# (0, 1) - (0.5, 1) - (1, 1)
#   
# (0, 0) ------------ (1, 0)

coords = np.array([[0, 0], [1, 0], [0, 1], [0.6, 1], [1, 1]])
elems = np.array([[0, 1, 2], [1, 2, 3], [1, 3, 4]])
test_element_assembler(coords, elems)
