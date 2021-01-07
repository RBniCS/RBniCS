# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import itertools
from sympy import Inverse, Matrix, MatrixSymbol, zeros
from rbnics.shape_parametrization.utils.symbolic.python_string_to_sympy import MatrixListSymbol, python_string_to_sympy
from rbnics.shape_parametrization.utils.symbolic.strings_to_sympy_symbolic_parameters import (
    strings_to_sympy_symbolic_parameters)
from rbnics.shape_parametrization.utils.symbolic.sympy_symbolic_coordinates import sympy_symbolic_coordinates


def affine_shape_parametrization_from_vertices_mapping(dim, vertices_mapping):
    # Check if the "identity" string is provided, and return this trivial case
    if isinstance(vertices_mapping, str):
        assert vertices_mapping.lower() == "identity"
        return tuple(["x[" + str(i) + "]" for i in range(dim)])
    # Get a sympy symbol for mu
    mu = strings_to_sympy_symbolic_parameters(itertools.chain(*vertices_mapping.values()), MatrixSymbol)
    # Convert vertices from string to symbols
    vertices_mapping_symbolic = dict()
    for (reference_vertex, deformed_vertex) in vertices_mapping.items():
        assert isinstance(reference_vertex, Matrix) == isinstance(deformed_vertex, Matrix)
        if isinstance(reference_vertex, Matrix):
            reference_vertex_symbolic = reference_vertex
            deformed_vertex_symbolic = deformed_vertex
        else:
            reference_vertex_symbolic = python_string_to_sympy(reference_vertex, None, None)
            deformed_vertex_symbolic = python_string_to_sympy(deformed_vertex, None, mu)
        assert reference_vertex_symbolic not in vertices_mapping_symbolic
        vertices_mapping_symbolic[reference_vertex_symbolic] = deformed_vertex_symbolic
    # Find A and b such that x_o = A x + b for all (x, x_o) in vertices_mapping
    lhs = zeros(dim + dim**2, dim + dim**2)
    rhs = zeros(dim + dim**2, 1)
    for (offset, (reference_vertex, deformed_vertex)) in enumerate(vertices_mapping_symbolic.items()):
        for i in range(dim):
            rhs[offset * dim + i] = deformed_vertex[i]
            lhs[offset * dim + i, i] = 1
            for j in range(dim):
                lhs[offset * dim + i, (i + 1) * dim + j] = reference_vertex[j]
    solution = Inverse(lhs) * rhs
    b = zeros(dim, 1)
    for i in range(dim):
        b[i] = solution[i]
    A = zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            A[i, j] = solution[dim + i * dim + j]
    # Convert into an expression
    x = sympy_symbolic_coordinates(dim, MatrixListSymbol)
    x_o = A * x + b
    return tuple([str(x_o[i]).replace(", 0]", "]") for i in range(dim)])
