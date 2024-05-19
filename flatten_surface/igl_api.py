import igl
import numpy as np

from .geometry import plane_through_3_points, rotate_points, rotation_matrix_from_vectors, \
    plane_normal_vector, find_boundary_loop


def init_unfold(vertices, faces, id_vertex):
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
    points = rotate_points(points, rotation_matrix_from_vectors(plane_normal_vector(plan), np.array([0, 0, 1])))
    points -= points[0]
    x, y, _ = points.T
    init_points_pos = np.ascontiguousarray(np.asarray([x, y], dtype=np.double).T)
    return init_points_ids, init_points_pos, plan


def unfold(vertices, faces, init_points_ids, init_points_pos):
    unwrap = igl.lscm(
        v=vertices,
        f=faces,
        b=init_points_ids,
        bc=init_points_pos,
    )[1]
    if not len(unwrap):
        raise Exception("Impossible to unfold")
    return unwrap


def get_all_bounds(faces):
    boundary_facets = igl.boundary_facets(faces)
    adjacency_list = {}
    for edge in boundary_facets:
        if edge[0] not in adjacency_list:
            adjacency_list[edge[0]] = []
        if edge[1] not in adjacency_list:
            adjacency_list[edge[1]] = []
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])
    all_boundary_loops = []
    visited = set()
    for edge in boundary_facets:
        if edge[0] not in visited:
            loop, adjacency_list = find_boundary_loop(edge[0], adjacency_list)
            all_boundary_loops.append(loop)
            visited.update(loop)
        if edge[1] not in visited:
            loop, adjacency_list = find_boundary_loop(edge[1], adjacency_list)
            all_boundary_loops.append(loop)
            visited.update(loop)
    return all_boundary_loops
