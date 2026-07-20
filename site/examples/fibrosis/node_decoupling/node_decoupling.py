
import matplotlib.pyplot as plt

import finitewave as fw
import numpy as np

# create a tissue of size 400x400 with cardiomycytes:
n, m = 30, 50
tissue = fw.CardiacTissue([n, m], dr=0.5)

y = np.arange(m//8, 3 * m//8)
x = np.full(len(y), n//2)
line = np.array([x, y]).T

tissue.add_pattern(fw.DecouplingPattern(line, axis=0))
tissue.add_pattern(fw.DecouplingPattern(density=0.4, axis=None,
                                        region=[[n//4, 3*n//4], [5*m//8, 7*m//8]]))

connectivity = tissue.connectivity
coords = np.argwhere(tissue.mesh == 1)

plt.figure()
plt.scatter(coords[:, 1], coords[:, 0], color="red", s=1)
plt.title("Removing connections by DecouplingPattern")
jj = np.arange(m)
for i in range(n-1):
    mask = connectivity[i, jj, 0] == 0
    x_ = [i + 0., i + 1.]
    y_ = jj[mask]
    y_ = [y_, y_]
    plt.plot(y_, x_, color="black", linewidth=1)

ii = np.arange(n)
for j in range(m-1):
    mask = connectivity[ii, j, 1] == 0
    x_ = ii[mask]
    x_ = [x_, x_]
    y_ = [j + 0., j + 1.]
    plt.plot(y_, x_, color="black", linewidth=1)
plt.show()
