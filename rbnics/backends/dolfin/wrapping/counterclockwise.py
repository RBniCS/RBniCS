# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Enum
X = 0
Y = 1


def counterclockwise(triangle_vertices):
    """
    mshr utility function to reorder triangle vertices in counterclockwise order
    """
    assert len(triangle_vertices) == 3
    triangle_vertices = list(triangle_vertices)
    assert all([len(coordinates) == 2 for coordinates in triangle_vertices])
    triangle_vertices_float = [[float(coord) for coord in vertex] for vertex in triangle_vertices]
    cross_product = (
        + (triangle_vertices_float[1][X] - triangle_vertices_float[0][X])
        * (triangle_vertices_float[2][Y] - triangle_vertices_float[0][Y])
        - (triangle_vertices_float[1][Y] - triangle_vertices_float[0][Y])
        * (triangle_vertices_float[2][X] - triangle_vertices_float[0][X])
    )
    if cross_product > 0:
        return triangle_vertices
    else:
        return [triangle_vertices[0], triangle_vertices[2], triangle_vertices[1]]
