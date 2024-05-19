import os
import tkinter as tk
from tkinter import filedialog

from .display import plot
from .igl_api import init_unfold, unfold, get_all_bounds
from .import_export import load, export_svg
from .score import compute_deformation


def main(path_stl=None, path_png=None, path_svg=None, vertice_init_id=0):
    while not os.path.isfile(path_stl):
        root = tk.Tk()
        root.withdraw()
        path_stl = filedialog.askopenfilename(
            initialdir=os.path.join(os.path.dirname(__file__), "..", "data"),
            title="Please select a STL file having surface (Not volume) to unfold",
            filetypes=(("STL surface", "*.stl"), ("STL surface", "*.STL"))
        )
    if path_png is None:
        path_png = os.path.join(os.path.dirname(path_stl), ".".join(os.path.basename(path_stl).split(".")[:-1]))
    if path_svg is None:
        path_svg = os.path.join(os.path.dirname(path_stl), ".".join(os.path.basename(path_stl).split(".")[:-1]) + ".svg")
    vertices, faces = load(path_stl)
    bounds = get_all_bounds(faces)
    init_points_ids, init_points_pos, plan = init_unfold(vertices, faces, vertice_init_id)
    unwrap = unfold(vertices, faces, init_points_ids, init_points_pos)
    deformation = compute_deformation(vertices, faces, unwrap)
    plot(vertices, faces, unwrap, init_points_pos, plan, init_points_ids, bounds, deformation, path_png)
    export_svg(unwrap, bounds, path_svg)
