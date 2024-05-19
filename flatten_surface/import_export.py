import numpy as np
import svgwrite
import trimesh


def load(path):
    mesh = trimesh.load_mesh(path)
    return (
        np.array(mesh.vertices, dtype=np.float64),
        np.array(mesh.faces, dtype=np.int64)
    )


def export_svg(unwrap, bounds, path_svg):
    contours = [unwrap[bound] for bound in bounds]

    min_x = float("inf")
    min_y = float("inf")

    for i, contour in enumerate(contours):
        x, y = contour.T
        min_x = min(min_x, np.min(x))
        min_y = min(min_y, np.min(y))

    for i, contour in enumerate(contours):
        x, y = contour.T
        x -= min_x
        y -= min_y
        contours[i] = np.array([x, y]).T

    dwg = svgwrite.Drawing(path_svg, profile='tiny')
    for contour in contours:
        path_data = "M" + " L".join(f"{x},{y}" for x, y in contour) + " Z"
        dwg.add(dwg.path(d=path_data, fill='none', stroke='black'))
    dwg.save()
