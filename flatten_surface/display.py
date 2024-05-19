import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot(vertices, faces, unwrap, init_points_pos, plan, init_points_ids, bounds, deformation, path_png):

    a, b, c, d = plan
    grid = np.meshgrid(
        np.linspace(min(vertices[:, 0]), max(vertices[:, 0]), 10),
        np.linspace(min(vertices[:, 1]), max(vertices[:, 1]), 10))

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, edgecolor='k', alpha=0.7)
    ax1.plot_surface(grid[0], grid[1], (d - a * grid[0] - b * grid[1]) / c, alpha=0.5, rstride=100, cstride=100)
    ax1.scatter(*vertices[init_points_ids].T, color="red")
    [ax1.plot(*vertices[bound].T, color="green") for bound in bounds]
    ax1.set_title('Original 3D Mesh')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.axis('equal')

    ax2 = fig.add_subplot(122)
    sc = ax2.tripcolor(unwrap[:, 0], unwrap[:, 1], faces, edgecolors='k', facecolors=deformation, cmap=cm.turbo)
    fig.colorbar(sc)
    ax2.scatter(*init_points_pos.T, color="red")
    [ax2.plot(*unwrap[bound].T, color="green") for bound in bounds]
    ax2.set_aspect('equal')
    ax2.set_title('Parameterized 2D Mesh (lscm)')
    ax2.set_xlabel('U')
    ax2.set_ylabel('V')
    ax2.axis('equal')

    plt.savefig(path_png, dpi=1000)
    plt.show()
