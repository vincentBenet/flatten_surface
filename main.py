import os

from flatten_surface import flatten_surface


if __name__ == "__main__":
    flatten_surface(os.path.join(os.path.dirname(__file__), "..", "data", 'eighth_of_a_sphere.STL_'))
