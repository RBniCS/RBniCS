# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
from sympy import Float, MatrixSymbol, preorder_traversal, simplify, sympify
from rbnics.shape_parametrization.utils.symbolic import (
    affine_shape_parametrization_from_vertices_mapping, VerticesMappingIO)

data_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data", "test_affine_shape_parametrization_from_vertices_mapping")

# Global enums for clearer notation for asserts
X = 0
Y = 1


# Check equality between symbolic expressions
def symbolic_equal(expression1, expression2, x, mu):
    locals = {"x": x, "mu": mu}
    difference = sympify(expression1, locals=locals) - sympify(expression2, locals=locals)
    difference = simplify(difference)
    for node in preorder_traversal(difference):
        if isinstance(node, Float):
            difference = difference.subs(node, round(node, 10))
    return difference == 0


# Test affine shape parametrization for tutorial 03
def test_affine_shape_parametrization_from_vertices_mapping_hole():
    filename = "vertices_mapping_hole"
    assert VerticesMappingIO.exists_file(data_dir, filename)
    vertices_mappings = VerticesMappingIO.load_file(data_dir, filename)
    shape_parametrization_expression = [
        affine_shape_parametrization_from_vertices_mapping(2, vertices_mapping)
        for vertices_mapping in vertices_mappings]
    # Auxiliary symbolic quantities
    x = MatrixSymbol("x", 2, 1)
    mu = MatrixSymbol("mu", 2, 1)
    # Start checks
    assert len(shape_parametrization_expression) == 8
    # Check subdomain 1
    assert len(shape_parametrization_expression[0]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[0][X], "2 - 2 * mu[0] + mu[0] * x[0] + (2 - 2 * mu[0]) * x[1]", x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[0][Y], "2 - 2 * mu[1] + (2 - mu[1]) * x[1]", x, mu)
    # Check subdomain 2
    assert len(shape_parametrization_expression[1]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[1][X], "2 * mu[0]- 2 + x[0] + (mu[0] - 1) * x[1]", x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[1][Y], "2 - 2 * mu[1] + (2 - mu[1]) * x[1]", x, mu)
    # Check subdomain 3
    assert len(shape_parametrization_expression[2]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[2][X], "2 - 2 * mu[0] + (2 - mu[0]) * x[0]", x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[2][Y], "2 - 2 * mu[1] + (2- 2*mu[1]) * x[0] + mu[1] * x[1]", x, mu)
    # Check subdomain 4
    assert len(shape_parametrization_expression[3]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[3][X], "2 - 2 * mu[0] + (2 - mu[0]) * x[0]", x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[3][Y], "2 * mu[1] - 2 + (mu[1] - 1) * x[0] + x[1]", x, mu)
    # Check subdomain 5
    assert len(shape_parametrization_expression[4]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[4][X], "2 * mu[0] - 2 + (2 - mu[0]) * x[0]", x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[4][Y], "2 - 2 * mu[1] + (2 * mu[1]- 2) * x[0] + mu[1] * x[1]", x, mu)
    # Check subdomain 6
    assert len(shape_parametrization_expression[5]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[5][X], "2 * mu[0] - 2 + (2 - mu[0]) * x[0]", x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[5][Y], "2 * mu[1] - 2 + (1 - mu[1]) * x[0] + x[1]", x, mu)
    # Check subdomain 7
    assert len(shape_parametrization_expression[6]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[6][X], "2 - 2 * mu[0] + mu[0] * x[0] + (2 * mu[0] - 2) * x[1]", x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[6][Y], "2 * mu[1] - 2 + (2 - mu[1]) * x[1]", x, mu)
    # Check subdomain 8
    assert len(shape_parametrization_expression[7]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[7][X], "2 * mu[0] - 2 + x[0] + (1 - mu[0]) * x[1]", x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[7][Y], "2 * mu[1] - 2 + (2 - mu[1]) * x[1]", x, mu)


# Test affine shape parametrization for tutorial 03 rotation
def test_affine_shape_parametrization_from_vertices_mapping_hole_rotation():
    vertices_mappings = [
        {
            ("-1", "-1"): ("-sqrt(2.0) * cos(mu[0])", "-sqrt(2.0) * sin(mu[0])"),
            ("-2", "-2"): ("-2", "-2"),
            ("1", "-1"): ("sqrt(2.0) * sin(mu[0])", "- sqrt(2.0) * cos(mu[0])")
        },  # subdomain 1
        {
            ("-2", "-2"): ("-2", "-2"),
            ("2", "-2"): ("2", "-2"),
            ("1", "-1"): ("sqrt(2.0) * sin(mu[0])", "- sqrt(2.0) * cos(mu[0])")
        },  # subdomain 2
        {
            ("-1", "-1"): ("- sqrt(2.0) * cos(mu[0])", "- sqrt(2.0) * sin(mu[0])"),
            ("-1", "1"): ("- sqrt(2.0) * sin(mu[0])", "sqrt(2.0) * cos(mu[0])"),
            ("-2", "-2"): ("-2", "-2")
        },  # subdomain 3
        {
            ("-1", "1"): ("- sqrt(2.0) * sin(mu[0])", "sqrt(2.0) * cos(mu[0])"),
            ("-2", "2"): ("-2", "2"),
            ("-2", "-2"): ("-2", "-2")
        },  # subdomain 4
        {
            ("1", "-1"): ("sqrt(2.0) * sin(mu[0])", "- sqrt(2.0) * cos(mu[0])"),
            ("2", "-2"): ("2", "-2"),
            ("1", "1"): ("sqrt(2.0) * cos(mu[0])", "sqrt(2.0) * sin(mu[0])")
        },  # subdomain 5
        {
            ("2", "2"): ("2", "2"),
            ("1", "1"): ("sqrt(2.0) * cos(mu[0])", "sqrt(2.0) * sin(mu[0])"),
            ("2", "-2"): ("2", "-2")
        },  # subdomain 6
        {
            ("-2", "2"): ("-2", "2"),
            ("-1", "1"): ("- sqrt(2.0) * sin(mu[0])", "sqrt(2.0) * cos(mu[0])"),
            ("1", "1"): ("sqrt(2.0) * cos(mu[0])", "sqrt(2.0) * sin(mu[0])")
        },  # subdomain 7
        {
            ("-2", "2"): ("-2", "2"),
            ("1", "1"): ("sqrt(2.0) * cos(mu[0])", "sqrt(2.0) * sin(mu[0])"),
            ("2", "2"): ("2", "2")
        }  # subdomain 8
    ]
    shape_parametrization_expression = [
        affine_shape_parametrization_from_vertices_mapping(2, vertices_mapping)
        for vertices_mapping in vertices_mappings]
    # Auxiliary symbolic quantities
    x = MatrixSymbol("x", 2, 1)
    mu = MatrixSymbol("mu", 1, 1)
    # Start checks
    assert len(shape_parametrization_expression) == 8
    # Check subdomain 1
    assert len(shape_parametrization_expression[0]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[0][X],
        "-2 * sqrt(2.0) * cos(mu[0]) + x[0] * (sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2)"
        + "+ x[1] * (- sqrt(2.0) * sin(mu[0]) / 2 - 3 * sqrt(2.0) * cos(mu[0]) / 2 + 2) + 2",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[0][Y],
        "-2 * sqrt(2.0) * sin(mu[0]) + x[0] * (sqrt(2.0) * sin(mu[0]) / 2 - sqrt(2.0) * cos(mu[0]) / 2)"
        + " + x[1] * (- 3 * sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2 + 2) + 2",
        x, mu)
    # Check subdomain 2
    assert len(shape_parametrization_expression[1]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[1][X],
        "2 * sqrt(2.0) * sin(mu[0]) + x[0] + x[1] * (sqrt(2.0) * sin(mu[0]) - 1) - 2",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[1][Y],
        "-2 * sqrt(2.0) * cos(mu[0]) + x[1] * (- sqrt(2.0) * cos(mu[0]) + 2) + 2",
        x, mu)
    # Check subdomain 3
    assert len(shape_parametrization_expression[2]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[2][X],
        "-2 * sqrt(2.0) * cos(mu[0]) + x[0] * (sqrt(2.0) * sin(mu[0]) / 2 - 3 * sqrt(2.0) * cos(mu[0]) / 2 + 2)"
        + "+ x[1] * (-sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2) + 2",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[2][Y],
        "-2 * sqrt(2.0) * sin(mu[0]) + x[0] * (- 3 * sqrt(2.0) * sin(mu[0]) / 2 - sqrt(2.0) * cos(mu[0]) / 2 + 2)"
        + "+ x[1] * (sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2) + 2",
        x, mu)
    # Check subdomain 4
    assert len(shape_parametrization_expression[3]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[3][X],
        "-2*sqrt(2.0) * sin(mu[0]) + x[0] * (-sqrt(2.0) * sin(mu[0]) + 2) + 2",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[3][Y],
        "2 * sqrt(2.0) * cos(mu[0]) + x[0] * (sqrt(2.0) * cos(mu[0]) - 1) + x[1] - 2",
        x, mu)
    # Check subdomain 5
    assert len(shape_parametrization_expression[4]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[4][X],
        "2 * sqrt(2.0) * sin(mu[0]) + x[0] * (-3 * sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2 + 2)"
        + "+ x[1] * (- sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2) - 2",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[4][Y],
        "-2 * sqrt(2.0) * cos(mu[0]) + x[0] * (sqrt(2.0) * sin(mu[0]) / 2 + 3 * sqrt(2.0) * cos(mu[0]) / 2 - 2)"
        + "+ x[1] * (sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2) + 2",
        x, mu)
    # Check subdomain 6
    assert len(shape_parametrization_expression[5]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[5][X],
        "2 * sqrt(2.0) * cos(mu[0]) + x[0] * (-sqrt(2.0) * cos(mu[0]) + 2) - 2",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[5][Y],
        "2 * sqrt(2.0) * sin(mu[0]) + x[0] * (-sqrt(2.0) * sin(mu[0]) + 1) + x[1] - 2",
        x, mu)
    # Check subdomain 7
    assert len(shape_parametrization_expression[6]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[6][X],
        "-2 * sqrt(2.0) * sin(mu[0]) + x[0] * (sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2)"
        + "+ x[1] * (3 * sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2 - 2) + 2",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[6][Y],
        "2 * sqrt(2.0) * cos(mu[0]) + x[0] * (sqrt(2.0) * sin(mu[0])/2 - sqrt(2.0)*cos(mu[0])/2)"
        + "+ x[1] * (sqrt(2.0) * sin(mu[0]) / 2 - 3 * sqrt(2.0) * cos(mu[0]) / 2 + 2) - 2",
        x, mu)
    # Check subdomain 8
    assert len(shape_parametrization_expression[7]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[7][X],
        "2 * sqrt(2.0) * cos(mu[0]) + x[0] + x[1] * (-sqrt(2.0) * cos(mu[0]) + 1) - 2",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[7][Y],
        "2 * sqrt(2.0) * sin(mu[0]) + x[1] * (-sqrt(2.0) * sin(mu[0]) + 2) - 2",
        x, mu)


# Test affine shape parametrization for tutorial 04
def test_affine_shape_parametrization_from_vertices_mapping_graetz():
    vertices_mappings = [
        {
            ("0", "0"): ("0", "0"),
            ("0", "1"): ("0", "1"),
            ("1", "1"): ("1", "1")
        },  # subdomain 1 top
        {
            ("0", "0"): ("0", "0"),
            ("1", "0"): ("1", "0"),
            ("1", "1"): ("1", "1")
        },  # subdomain 1 bottom
        {
            ("1", "0"): ("1", "0"),
            ("1", "1"): ("1", "1"),
            ("2", "1"): ("1 + mu[0]", "1")
        },  # subdomain 2 top
        {
            ("1", "0"): ("1", "0"),
            ("2", "0"): ("1 + mu[0]", "0"),
            ("2", "1"): ("1 + mu[0]", "1")
        }  # subdomain 2 bottom
    ]
    shape_parametrization_expression = [
        affine_shape_parametrization_from_vertices_mapping(2, vertices_mapping)
        for vertices_mapping in vertices_mappings]
    # Auxiliary symbolic quantities
    x = MatrixSymbol("x", 2, 1)
    mu = MatrixSymbol("mu", 1, 1)
    # Start checks
    assert len(shape_parametrization_expression) == 4
    # Check subdomain 1 top
    assert len(shape_parametrization_expression[0]) == 2
    assert symbolic_equal(shape_parametrization_expression[0][X], "x[0]", x, mu)
    assert symbolic_equal(shape_parametrization_expression[0][Y], "x[1]", x, mu)
    # Check subdomain 1 bottom
    assert len(shape_parametrization_expression[1]) == 2
    assert symbolic_equal(shape_parametrization_expression[1][X], "x[0]", x, mu)
    assert symbolic_equal(shape_parametrization_expression[1][Y], "x[1]", x, mu)
    # Check subdomain 2 top
    assert len(shape_parametrization_expression[2]) == 2
    assert symbolic_equal(shape_parametrization_expression[2][X], "mu[0] * (x[0] - 1) + 1", x, mu)
    assert symbolic_equal(shape_parametrization_expression[2][Y], "x[1]", x, mu)
    # Check subdomain 2 bottom
    assert len(shape_parametrization_expression[3]) == 2
    assert symbolic_equal(shape_parametrization_expression[3][X], "mu[0] * (x[0] - 1) + 1", x, mu)
    assert symbolic_equal(shape_parametrization_expression[3][Y], "x[1]", x, mu)


# Test affine shape parametrization for tutorial 12
def test_affine_shape_parametrization_from_vertices_mapping_stokes():
    filename = "vertices_mapping_stokes"
    assert VerticesMappingIO.exists_file(data_dir, filename)
    vertices_mappings = VerticesMappingIO.load_file(data_dir, filename)
    shape_parametrization_expression = [
        affine_shape_parametrization_from_vertices_mapping(2, vertices_mapping)
        for vertices_mapping in vertices_mappings]
    # Auxiliary symbolic quantities
    x = MatrixSymbol("x", 2, 1)
    mu = MatrixSymbol("mu", 2, 6)
    # Start checks
    assert len(shape_parametrization_expression) == 8
    # Check subdomain 1
    assert len(shape_parametrization_expression[0]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[0][X],
        "mu[4] * x[0] + mu[1] - mu[4]",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[0][Y],
        "mu[4] * tan(mu[5]) * x[0] + mu[0] * x[1] + mu[2] - mu[4] * tan(mu[5]) - mu[0]",
        x, mu)
    # Check subdomain 2
    assert len(shape_parametrization_expression[1]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[1][X],
        "mu[4] * x[0] + mu[1] - mu[4]",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[1][Y],
        "mu[4] * tan(mu[5]) * x[0] + mu[0] * x[1] + mu[2] - mu[4] * tan(mu[5]) - mu[0]",
        x, mu)
    # Check subdomain 3
    assert len(shape_parametrization_expression[2]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[2][X],
        "mu[1] * x[0]",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[2][Y],
        "mu[3] * x[1] + mu[2] + mu[0] - 2 * mu[3]",
        x, mu)
    # Check subdomain 4
    assert len(shape_parametrization_expression[3]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[3][X],
        "mu[1] * x[0]",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[3][Y],
        "mu[3] * x[1] + mu[2] + mu[0] - 2 * mu[3]",
        x, mu)
    # Check subdomain 5
    assert len(shape_parametrization_expression[4]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[4][X],
        "mu[1] * x[0]",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[4][Y],
        "mu[0] * x[1] + mu[2] - mu[0]",
        x, mu)
    # Check subdomain 6
    assert len(shape_parametrization_expression[5]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[5][X],
        "mu[1] * x[0]",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[5][Y],
        "mu[0] * x[1] + mu[2] - mu[0]",
        x, mu)
    # Check subdomain 7
    assert len(shape_parametrization_expression[6]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[6][X],
        "mu[1] * x[0]",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[6][Y],
        "mu[2] * x[1]",
        x, mu)
    # Check subdomain 8
    assert len(shape_parametrization_expression[7]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[7][X],
        "mu[1] * x[0]",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[7][Y],
        "mu[2] * x[1]",
        x, mu)


# Test affine shape parametrization for tutorial 17
def test_affine_shape_parametrization_from_vertices_mapping_navier_stokes():
    vertices_mappings = [
        {
            ("0.0", "5.0"): ("0.0", "5.0"),
            ("0.0", "2.0"): ("0.0", "2.0"),
            ("22.0", "2.0"): ("22.0", "2.0")
        },  # subdomain 1 bottom
        {
            ("22.0", "2.0"): ("22.0", "2.0"),
            ("22.0", "5.0"): ("22.0", "5.0"),
            ("0.0", "5.0"): ("0.0", "5.0")
        },  # subdomain 1 top
        {
            ("4.0", "2.0"): ("4.0", "2.0"),
            ("4.0", "0.0"): ("4.0", "2.0 - mu[1]"),
            ("22.0", "0.0"): ("22.0", "2.0 - mu[1]")
        },  # subdomain 2 bottom
        {
            ("22.0", "0.0"): ("22.0", "2.0 - mu[1]"),
            ("22.0", "2.0"): ("22.0", "2.0"),
            ("4.0", "2.0"): ("4.0", "2.0")
        }  # subdomain 2 top
    ]
    shape_parametrization_expression = [
        affine_shape_parametrization_from_vertices_mapping(2, vertices_mapping)
        for vertices_mapping in vertices_mappings]
    # Auxiliary symbolic quantities
    x = MatrixSymbol("x", 2, 1)
    mu = MatrixSymbol("mu", 2, 1)
    # Start checks
    assert len(shape_parametrization_expression) == 4
    # Check subdomain 1 top
    assert len(shape_parametrization_expression[0]) == 2
    assert symbolic_equal(shape_parametrization_expression[0][X], "x[0]", x, mu)
    assert symbolic_equal(shape_parametrization_expression[0][Y], "x[1]", x, mu)
    # Check subdomain 1 bottom
    assert len(shape_parametrization_expression[1]) == 2
    assert symbolic_equal(shape_parametrization_expression[1][X], "x[0]", x, mu)
    assert symbolic_equal(shape_parametrization_expression[1][Y], "x[1]", x, mu)
    # Check subdomain 2 top
    assert len(shape_parametrization_expression[2]) == 2
    assert symbolic_equal(shape_parametrization_expression[2][X], "x[0]", x, mu)
    assert symbolic_equal(shape_parametrization_expression[2][Y], "0.5 * mu[1] * x[1] - 1.0 * mu[1] + 2.0", x, mu)
    # Check subdomain 2 bottom
    assert len(shape_parametrization_expression[3]) == 2
    assert symbolic_equal(shape_parametrization_expression[3][X], "x[0]", x, mu)
    assert symbolic_equal(shape_parametrization_expression[3][Y], "0.5 * mu[1] * x[1] - 1.0 * mu[1] + 2.0", x, mu)


# Test affine shape parametrization for stokes optimal dirichlet boundary control
def test_affine_shape_parametrization_from_vertices_mapping_stokes_optimal_dirichlet_boundary_control():
    vertices_mappings = [
        "identity",  # subdomain 1
        {
            ("0.9", "0.0"): ("0.9", "0.0"),
            ("1.0", "0.0"): ("0.9 + mu[0]", "0.0"),
            ("0.9", "0.4"): ("0.9", "0.4")
        },  # subdomain 2
        {
            ("1.0", "0.0"): ("0.9 + mu[0]", "0.0"),
            ("1.0", "0.4"): ("0.9 + mu[0]", "0.4"),
            ("0.9", "0.4"): ("0.9", "0.4")
        },  # subdomain 3
        {
            ("0.9", "0.6"): ("0.9", "0.6"),
            ("1.0", "0.6"): ("0.9 + mu[0]", "0.6"),
            ("0.9", "1.0"): ("0.9", "1.0")
        },  # subdomain 4
        {
            ("1.0", "0.6"): ("0.9 + mu[0]", "0.6"),
            ("1.0", "1.0"): ("0.9 + mu[0]", "1.0"),
            ("0.9", "1.0"): ("0.9", "1.0")
        },  # subdomain 5
        {
            ("1.0", "0.0"): ("0.9 + mu[0]", "0.0"),
            ("1.8", "0.2"): ("1.8", "0.2"),
            ("1.0", "0.4"): ("0.9 + mu[0]", "0.4")
        },  # subdomain 6
        {
            ("1.0", "0.0"): ("0.9 + mu[0]", "0.0"),
            ("2.0", "0.0"): ("2.0", "0.0"),
            ("1.8", "0.2"): ("1.8", "0.2")
        },  # subdomain 7
        {
            ("1.0", "0.6"): ("0.9 + mu[0]", "0.6"),
            ("1.8", "0.8"): ("1.8", "0.8"),
            ("1.0", "1.0"): ("0.9 + mu[0]", "1.0")
        },  # subdomain 8
        {
            ("1.0", "1.0"): ("0.9 + mu[0]", "1.0"),
            ("1.8", "0.8"): ("1.8", "0.8"),
            ("2.0", "1.0"): ("2.0", "1.0")
        },  # subdomain 9
        {
            ("1.8", "0.8"): ("1.8", "0.8"),
            ("2.0", "0.0"): ("2.0", "0.0"),
            ("2.0", "1.0"): ("2.0", "1.0")
        },  # subdomain 10
        {
            ("1.8", "0.8"): ("1.8", "0.8"),
            ("1.8", "0.2"): ("1.8", "0.2"),
            ("2.0", "0.0"): ("2.0", "0.0")
        },  # subdomain 11
        {
            ("1.0", "0.4"): ("0.9 + mu[0]", "0.4"),
            ("1.8", "0.2"): ("1.8", "0.2"),
            ("1.0", "0.6"): ("0.9 + mu[0]", "0.6")
        },  # subdomain 12
        {
            ("1.0", "0.6"): ("0.9 + mu[0]", "0.6"),
            ("1.8", "0.2"): ("1.8", "0.2"),
            ("1.8", "0.8"): ("1.8", "0.8")
        }  # subdomain 13
    ]
    shape_parametrization_expression = [
        affine_shape_parametrization_from_vertices_mapping(2, vertices_mapping)
        for vertices_mapping in vertices_mappings]
    # Auxiliary symbolic quantities
    x = MatrixSymbol("x", 2, 1)
    mu = MatrixSymbol("mu", 1, 1)
    # Start checks
    assert len(shape_parametrization_expression) == 13
    # Check subdomain 1
    assert len(shape_parametrization_expression[0]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[0][X],
        "x[0]",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[0][Y],
        "x[1]",
        x, mu)
    # Check subdomain 2
    assert len(shape_parametrization_expression[1]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[1][X],
        "0.9 - 9.0 * mu[0] + 10.0 * mu[0] * x[0]",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[1][Y],
        "x[1]",
        x, mu)
    # Check subdomain 3
    assert len(shape_parametrization_expression[2]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[2][X],
        "0.9 - 9.0 * mu[0] + 10.0 * mu[0] * x[0]",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[2][Y],
        "x[1]",
        x, mu)
    # Check subdomain 4
    assert len(shape_parametrization_expression[3]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[3][X],
        "0.9 - 9.0 * mu[0] + 10.0 * mu[0] * x[0]",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[3][Y],
        "x[1]",
        x, mu)
    # Check subdomain 5
    assert len(shape_parametrization_expression[4]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[4][X],
        "0.9 - 9.0 * mu[0] + 10.0 * mu[0] * x[0]",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[4][Y],
        "x[1]",
        x, mu)
    # Check subdomain 6
    assert len(shape_parametrization_expression[5]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[5][X],
        "2.25 * mu[0] + x[0] * (-1.25 * mu[0] + 1.125) - 0.225",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[5][Y],
        "x[1]",
        x, mu)
    # Check subdomain 7
    assert len(shape_parametrization_expression[6]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[6][X],
        "2.0 * mu[0] + x[0] * (-mu[0] + 1.1) + x[1] * (-mu[0] + 0.1) - 0.2",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[6][Y],
        "x[1]",
        x, mu)
    # Check subdomain 8
    assert len(shape_parametrization_expression[7]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[7][X],
        "2.25 * mu[0] + x[0] * (-1.25 * mu[0] + 1.125) - 0.225",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[7][Y],
        "x[1]",
        x, mu)
    # Check subdomain 9
    assert len(shape_parametrization_expression[8]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[8][X],
        "mu[0] + x[0] * (-mu[0] + 1.1) + x[1] * (mu[0] - 0.1) - 0.1",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[8][Y],
        "x[1]",
        x, mu)
    # Check subdomain 10
    assert len(shape_parametrization_expression[9]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[9][X],
        "x[0]",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[9][Y],
        "x[1]",
        x, mu)
    # Check subdomain 11
    assert len(shape_parametrization_expression[10]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[10][X],
        "x[0]",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[10][Y],
        "x[1]",
        x, mu)
    # Check subdomain 12
    assert len(shape_parametrization_expression[11]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[11][X],
        "x[0] * (-1.25 * mu[0] + 1.125) + 2.25 * mu[0] - 0.225",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[11][Y],
        "x[1]",
        x, mu)
    # Check subdomain 13
    assert len(shape_parametrization_expression[12]) == 2
    assert symbolic_equal(
        shape_parametrization_expression[12][X],
        "2.25 * mu[0] + x[0] * (-1.25 * mu[0] + 1.125) - 0.225",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_expression[12][Y],
        "x[1]",
        x, mu)
