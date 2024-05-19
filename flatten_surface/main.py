import os

from flatten_surface.display import plot
from flatten_surface.igl_api import init_unfold, unfold, get_all_bounds
from flatten_surface.import_export import load, export_svg
from flatten_surface.score import compute_deformation


def main(path):
    vertices, faces = load(path)
    bounds = get_all_bounds(faces)
    init_points_ids, init_points_pos, plan = init_unfold(vertices, faces)
    unwrap = unfold(vertices, faces, init_points_ids, init_points_pos)
    deformation = compute_deformation(vertices, faces, unwrap)
    plot(vertices, faces, unwrap, init_points_pos, plan, init_points_ids, bounds, deformation, path)
    export_svg(unwrap, bounds, path)


if __name__ == "__main__":
    main(os.path.join(os.path.dirname(__file__), "..", "data", 'eighth_of_a_sphere.STL'))
