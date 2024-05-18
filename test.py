import math
import os
import igl
import numpy
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def angle_between_3_points(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    vec_ab = np.array([x2 - x1, y2 - y1, z2 - z1])
    vec_bc = np.array([x3 - x2, y3 - y2, z3 - z2])
    dot_product = np.dot(vec_ab, vec_bc)
    magnitudes_product = np.linalg.norm(vec_ab) * np.linalg.norm(vec_bc)
    angle_rad = np.arccos(dot_product / magnitudes_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def plane_through_3_points(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    vec1 = np.array([x2 - x1, y2 - y1, z2 - z1])
    vec2 = np.array([x3 - x1, y3 - y1, z3 - z1])
    normal_vector = np.cross(vec1, vec2)
    a, b, c = normal_vector
    d = a*x1 + b*y1 + c*z1
    return a, b, c, d


def rotation_matrix_from_vectors(vec1, vec2):
    # Find the rotation axis using cross product
    axis = np.cross(vec1, vec2)
    axis = axis / np.linalg.norm(axis)

    # Find the angle between the vectors
    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    # Compute the components of the rotation matrix
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    one_minus_cos = 1.0 - cos_angle

    ux, uy, uz = axis

    rotation_matrix = np.array([
        [cos_angle + ux*ux*one_minus_cos, ux*uy*one_minus_cos - uz*sin_angle, ux*uz*one_minus_cos + uy*sin_angle],
        [uy*ux*one_minus_cos + uz*sin_angle, cos_angle + uy*uy*one_minus_cos, uy*uz*one_minus_cos - ux*sin_angle],
        [uz*ux*one_minus_cos - uy*sin_angle, uz*uy*one_minus_cos + ux*sin_angle, cos_angle + uz*uz*one_minus_cos]
    ])

    return rotation_matrix


def rotate_points(points, rotation_matrix):
    return np.dot(points, rotation_matrix.T)


def plane_normal_vector(plan):
    normal_vector = np.array([plan[0], plan[1], plan[2]])
    return normal_vector / np.linalg.norm(normal_vector)


def load(path):
    mesh = trimesh.load_mesh(path)
    return (
        np.array(mesh.vertices, dtype=np.float64),
        np.array(mesh.faces, dtype=np.int64)
    )


def init_unfold(vertices, faces, id_vertex=None):
    if id_vertex is None:
        id_vertex = 0
    init_points_ids = np.array(faces[id_vertex], dtype=np.int64)
    x1 = vertices[init_points_ids[0]][0]
    y1 = vertices[init_points_ids[0]][1]
    z1 = vertices[init_points_ids[0]][2]
    x2 = vertices[init_points_ids[1]][0]
    y2 = vertices[init_points_ids[1]][1]
    z2 = vertices[init_points_ids[1]][2]
    x3 = vertices[init_points_ids[2]][0]
    y3 = vertices[init_points_ids[2]][1]
    z3 = vertices[init_points_ids[2]][2]
    points = np.array([
        [x1, y1, z1],
        [x2, y2, z2],
        [x3, y3, z3]
    ])
    plan = plane_through_3_points(x1, y1, z1, x2, y2, z2, x3, y3, z3)
    points = rotate_points(points, rotation_matrix_from_vectors(plane_normal_vector(plan), numpy.array([0, 0, 1])))
    points -= points[0]
    x, y, _ = points.T
    init_points_pos = np.ascontiguousarray(numpy.asarray([x, y], dtype=np.double).T)
    return init_points_ids, init_points_pos, plan


def unfold(vertices, faces, init_points_ids, init_points_pos):
    unfolded = igl.lscm(
        v=vertices,
        f=faces,
        b=init_points_ids,
        bc=init_points_pos,
    )[1]
    if not len(unfolded):
        raise Exception("Impossible to unfold")
    return unfolded


def get_bounds(faces):
    return igl.boundary_loop(faces)


def compute_deformation(vertices, faces, unfolded):
    original_edges = vertices[faces[:, [1, 2, 0]]] - vertices[faces[:, [0, 1, 2]]]
    unfolded_edges = unfolded[faces[:, [1, 2, 0]]] - unfolded[faces[:, [0, 1, 2]]]

    original_lengths = np.linalg.norm(original_edges, axis=2)
    unfolded_lengths = np.linalg.norm(unfolded_edges, axis=2)

    length_ratios = unfolded_lengths / original_lengths

    original_angles = np.arccos(np.clip(np.sum(original_edges[:, 0] * original_edges[:, 1], axis=1) /
                                        (np.linalg.norm(original_edges[:, 0], axis=1) * np.linalg.norm(
                                            original_edges[:, 1], axis=1)), -1.0, 1.0))
    unfolded_angles = np.arccos(np.clip(np.sum(unfolded_edges[:, 0] * unfolded_edges[:, 1], axis=1) /
                                        (np.linalg.norm(unfolded_edges[:, 0], axis=1) * np.linalg.norm(
                                            unfolded_edges[:, 1], axis=1)), -1.0, 1.0))

    original_areas = np.linalg.norm(np.cross(original_edges[:, 0], original_edges[:, 1]), axis=1)
    unfolded_areas = np.cross(unfolded_edges[:, 0], unfolded_edges[:, 1])

    area_2d = np.sum(unfolded_areas)
    area_3d = np.sum(original_areas)

    area_diff = area_3d - area_2d

    print(f"3D Area: {area_3d} mm²")
    print(f"2D Area: {area_2d} mm²")
    print(f"Diff Area: {area_diff} mm²")

    area_diff = original_areas - unfolded_areas
    angle_diff = unfolded_angles - original_angles
    length_diff = np.linalg.norm(np.abs(length_ratios - 1), axis=1)

    area_distortion = numpy.abs(area_diff)
    angle_distortion = numpy.abs(angle_diff)
    length_distortion = numpy.abs(length_diff)

    area_lin = (area_distortion - min(area_distortion)) / (max(area_distortion) - min(area_distortion))
    angle_lin = (angle_distortion - min(angle_distortion)) / (max(angle_distortion) - min(angle_distortion))
    length_lin = (length_distortion - min(length_distortion)) / (max(length_distortion) - min(length_distortion))

    distortion = area_lin * angle_lin * length_lin
    distortion = 100 * (distortion - min(distortion)) / (max(distortion) - min(distortion))

    return area_diff


def plot(vertices, faces, unfolded, init_points_pos, plan, init_points_ids, bounds, deformation):

    a, b, c, d = plan
    grid = np.meshgrid(
        np.linspace(min(vertices[:, 0]), max(vertices[:, 0]), 10),
        np.linspace(min(vertices[:, 1]), max(vertices[:, 1]), 10))

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=F, edgecolor='k', alpha=0.7)
    ax1.plot_surface(grid[0], grid[1], (d - a * grid[0] - b * grid[1]) / c, alpha=0.5, rstride=100, cstride=100)
    ax1.scatter(*vertices[init_points_ids].T, color="red")
    ax1.plot(*vertices[bounds].T, color="green")
    ax1.set_title('Original 3D Mesh')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.axis('equal')

    ax2 = fig.add_subplot(122)
    sc = ax2.tripcolor(unfolded[:, 0], unfolded[:, 1], faces, edgecolors='k', facecolors=deformation, cmap=cm.turbo)
    fig.colorbar(sc)
    ax2.scatter(*init_points_pos.T, color="red")
    ax2.plot(*unfolded[bounds].T, color="green")
    ax2.set_aspect('equal')
    ax2.set_title('Parameterized 2D Mesh (lscm)')
    ax2.set_xlabel('U')
    ax2.set_ylabel('V')
    ax2.axis('equal')

    plt.show()


if __name__ == "__main__":
    V, F = load(os.path.join(os.path.dirname(__file__), 'sphere_sur_8_trou.STL'))
    bnd = get_bounds(F)
    b, bc, p = init_unfold(V, F)
    U = unfold(V, F, b, bc)
    deformation = compute_deformation(V, F, U)
    plot(V, F, U, bc, p, b, bnd, deformation)
