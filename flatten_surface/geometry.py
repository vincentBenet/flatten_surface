import numpy as np


def plane_through_3_points(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    a, b, c = np.cross(np.array([x2 - x1, y2 - y1, z2 - z1]), np.array([x3 - x1, y3 - y1, z3 - z1]))
    d = a*x1 + b*y1 + c*z1
    return a, b, c, d


def rotation_matrix_from_vectors(vec1, vec2):
    axis = np.cross(vec1, vec2)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    one_minus_cos = 1.0 - cos_angle
    ux, uy, uz = axis
    return np.array([
        [cos_angle + ux*ux*one_minus_cos, ux*uy*one_minus_cos - uz*sin_angle, ux*uz*one_minus_cos + uy*sin_angle],
        [uy*ux*one_minus_cos + uz*sin_angle, cos_angle + uy*uy*one_minus_cos, uy*uz*one_minus_cos - ux*sin_angle],
        [uz*ux*one_minus_cos - uy*sin_angle, uz*uy*one_minus_cos + ux*sin_angle, cos_angle + uz*uz*one_minus_cos]])


def rotate_points(points, rotation_matrix):
    return np.dot(points, rotation_matrix.T)


def plane_normal_vector(plan):
    normal_vector = np.array([plan[0], plan[1], plan[2]])
    return normal_vector / np.linalg.norm(normal_vector)


def find_boundary_loop(start_vertex, adjacency_list):
    boundary_loop = []
    current_vertex = start_vertex
    previous_vertex = None
    while True:
        boundary_loop.append(current_vertex)
        neighbors = adjacency_list[current_vertex]
        next_vertex = neighbors[0] if neighbors[0] != previous_vertex else neighbors[1]
        if next_vertex == start_vertex:
            break
        previous_vertex = current_vertex
        current_vertex = next_vertex
    return boundary_loop, adjacency_list
