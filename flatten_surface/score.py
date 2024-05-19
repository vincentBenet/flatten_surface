import numpy as np


def compute_deformation(vertices, faces, unwrap):
    original_edges = vertices[faces[:, [1, 2, 0]]] - vertices[faces[:, [0, 1, 2]]]
    unfolded_edges = unwrap[faces[:, [1, 2, 0]]] - unwrap[faces[:, [0, 1, 2]]]

    # original_lengths = np.linalg.norm(original_edges, axis=2)
    # unfolded_lengths = np.linalg.norm(unfolded_edges, axis=2)

    # length_ratios = unfolded_lengths / original_lengths
    #
    # original_angles = np.arccos(np.clip(np.sum(original_edges[:, 0] * original_edges[:, 1], axis=1) /
    #                                     (np.linalg.norm(original_edges[:, 0], axis=1) * np.linalg.norm(
    #                                         original_edges[:, 1], axis=1)), -1.0, 1.0))
    # unfolded_angles = np.arccos(np.clip(np.sum(unfolded_edges[:, 0] * unfolded_edges[:, 1], axis=1) /
    #                                     (np.linalg.norm(unfolded_edges[:, 0], axis=1) * np.linalg.norm(
    #                                         unfolded_edges[:, 1], axis=1)), -1.0, 1.0))

    original_areas = np.linalg.norm(np.cross(original_edges[:, 0], original_edges[:, 1]), axis=1)
    unfolded_areas = np.cross(unfolded_edges[:, 0], unfolded_edges[:, 1])
    area_2d = np.sum(unfolded_areas)
    area_3d = np.sum(original_areas)

    area_diff = area_3d - area_2d

    print(f"3D Area: {area_3d} mm²")
    print(f"2D Area: {area_2d} mm²")
    print(f"Diff Area: {area_diff} mm²")

    area_diff = original_areas - unfolded_areas
    # angle_diff = unfolded_angles - original_angles
    # length_diff = np.linalg.norm(np.abs(length_ratios - 1), axis=1)

    area_distortion = np.abs(area_diff)
    # angle_distortion = numpy.abs(angle_diff)
    # length_distortion = numpy.abs(length_diff)

    # area_lin = (area_distortion - min(area_distortion)) / (max(area_distortion) - min(area_distortion))
    # angle_lin = (angle_distortion - min(angle_distortion)) / (max(angle_distortion) - min(angle_distortion))
    # length_lin = (length_distortion - min(length_distortion)) / (max(length_distortion) - min(length_distortion))

    # distortion = area_lin * angle_lin * length_lin
    # distortion = 100 * (distortion - min(distortion)) / (max(distortion) - min(distortion))

    return area_distortion
