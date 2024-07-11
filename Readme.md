# Flatten Surface
This script execute similar behavior as premium feature "Flattening Surfaces" of solidworks: https://help.solidworks.com/2021/english/solidworks/sldworks/t_flattening_surfaces.htm

## Inputs
The input is a file path as a surface into a .STL file.

The result quality depends on the meshing quality!

## Outputs
The output is a .SVG file of the flattened surface.

A graph display area deformation foreach vertices.

## Operation
This project has following steps:
	- Load STL file as a mesh with trimesh: https://trimesh.org/
	- Compute all bounds with a faces analysis, based on the function boundary_loop from igl librairy: https://libigl.github.io/libigl-python-bindings/igl_docs/#boundary_loop
	- Select a vertice and rotate it on a XY plan as initialisation vertice using chatGPT 4.O
	- Mesh unfolding using the function lscm from igl librairy: https://libigl.github.io/libigl-python-bindings/igl_docs/#lscm
	- Deformation compting on area using chatGPT 4.O
	- Display the result with 3D surface, flattened surface, bounds and deformations heatmap with matplotlib tripcolor function: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.tripcolor.html
	- Export result as SVG file using svgwrite librairy: https://svgwrite.readthedocs.io/en/latest/

## Usage
	To use the project, call the function flatten_surface.main.main with the argument of your STL file containing a surface
