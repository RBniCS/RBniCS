# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from sympy import MatrixSymbol
from rbnics.shape_parametrization.utils.symbolic import compute_shape_parametrization_gradient
from test_affine_shape_parametrization_from_vertices_mapping import symbolic_equal, X, Y


# Test shape parametrization gradient computation for tutorial 03
def test_compute_shape_parametrization_gradient_hole():
    shape_parametrization_expression = [
        ("2 - 2 * mu[0] + mu[0] * x[0] + (2 - 2 * mu[0]) * x[1]", "2 - 2 * mu[1] + (2 - mu[1]) * x[1]"),  # subdomain 1
        ("2 * mu[0]- 2 + x[0] + (mu[0] - 1) * x[1]", "2 - 2 * mu[1] + (2 - mu[1]) * x[1]"),  # subdomain 2
        ("2 - 2 * mu[0] + (2 - mu[0]) * x[0]", "2 - 2 * mu[1] + (2 - 2 * mu[1]) * x[0] + mu[1] * x[1]"),  # subdomain 3
        ("2 - 2 * mu[0] + (2 - mu[0]) * x[0]", "2 * mu[1] - 2 + (mu[1] - 1) * x[0] + x[1]"),  # subdomain 4
        ("2 * mu[0] - 2 + (2 - mu[0]) * x[0]", "2 - 2 * mu[1] + (2 * mu[1]- 2) * x[0] + mu[1] * x[1]"),  # subdomain 5
        ("2 * mu[0] - 2 + (2 - mu[0]) * x[0]", "2 * mu[1] - 2 + (1 - mu[1]) * x[0] + x[1]"),  # subdomain 6
        ("2 - 2 * mu[0] + mu[0] * x[0] + (2 * mu[0] - 2) * x[1]", "2 * mu[1] - 2 + (2 - mu[1]) * x[1]"),  # subdomain 7
        ("2 * mu[0] - 2 + x[0] + (1 - mu[0]) * x[1]", "2 * mu[1] - 2 + (2 - mu[1]) * x[1]")  # subdomain 8
    ]
    shape_parametrization_gradient_expression = [
        compute_shape_parametrization_gradient(expression_on_subdomain)
        for expression_on_subdomain in shape_parametrization_expression]
    # Auxiliary symbolic quantities
    x = MatrixSymbol("x", 2, 1)
    mu = MatrixSymbol("mu", 2, 1)
    # Start checks
    assert len(shape_parametrization_gradient_expression) == 8
    # Check subdomain 1
    assert len(shape_parametrization_gradient_expression[0]) == 2
    assert len(shape_parametrization_gradient_expression[0][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[0][X][X], "mu[0]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[0][X][Y], "2 - 2 * mu[0]", x, mu)
    assert len(shape_parametrization_gradient_expression[0][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[0][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[0][Y][Y], "2 - mu[1]", x, mu)
    # Check subdomain 2
    assert len(shape_parametrization_gradient_expression[1]) == 2
    assert len(shape_parametrization_gradient_expression[1][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[1][X][X], "1", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[1][X][Y], "mu[0] - 1", x, mu)
    assert len(shape_parametrization_gradient_expression[1][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[1][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[1][Y][Y], "2 - mu[1]", x, mu)
    # Check subdomain 3
    assert len(shape_parametrization_gradient_expression[2]) == 2
    assert len(shape_parametrization_gradient_expression[2][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[2][X][X], "2 - mu[0]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[2][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[2][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[2][Y][X], "2 - 2 * mu[1]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[2][Y][Y], "mu[1]", x, mu)
    # Check subdomain 4
    assert len(shape_parametrization_gradient_expression[3]) == 2
    assert len(shape_parametrization_gradient_expression[3][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[3][X][X], "2 - mu[0]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[3][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[3][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[3][Y][X], "mu[1] - 1", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[3][Y][Y], "1", x, mu)
    # Check subdomain 5
    assert len(shape_parametrization_gradient_expression[4]) == 2
    assert len(shape_parametrization_gradient_expression[4][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[4][X][X], "2 - mu[0]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[4][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[4][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[4][Y][X], "2 * mu[1]- 2", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[4][Y][Y], "mu[1]", x, mu)
    # Check subdomain 6
    assert len(shape_parametrization_gradient_expression[5]) == 2
    assert len(shape_parametrization_gradient_expression[5][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[5][X][X], "2 - mu[0]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[5][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[5][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[5][Y][X], "1 - mu[1]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[5][Y][Y], "1", x, mu)
    # Check subdomain 7
    assert len(shape_parametrization_gradient_expression[6]) == 2
    assert len(shape_parametrization_gradient_expression[6][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[6][X][X], "mu[0]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[6][X][Y], "2 * mu[0] - 2", x, mu)
    assert len(shape_parametrization_gradient_expression[6][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[6][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[6][Y][Y], "2 - mu[1]", x, mu)
    # Check subdomain 8
    assert len(shape_parametrization_gradient_expression[7]) == 2
    assert len(shape_parametrization_gradient_expression[7][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[7][X][X], "1", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[7][X][Y], "1 - mu[0]", x, mu)
    assert len(shape_parametrization_gradient_expression[7][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[7][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[7][Y][Y], "2 - mu[1]", x, mu)


# Test shape parametrization gradient computation for tutorial 03 rotation
def test_compute_shape_parametrization_gradient_hole_rotation():
    shape_parametrization_expression = [
        ("- 2 * sqrt(2.0) * cos(mu[0]) + x[0] * (sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2)"
         + "+ x[1] * (- sqrt(2.0) * sin(mu[0]) / 2 - 3 * sqrt(2.0) * cos(mu[0]) / 2 + 2) + 2",
         "- 2 * sqrt(2.0) * sin(mu[0]) + x[0] * (sqrt(2.0) * sin(mu[0]) / 2 - sqrt(2.0) * cos(mu[0]) / 2)"
         + "+ x[1] * (- 3 * sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2 + 2) + 2"),  # subdomain 1
        ("2 * sqrt(2.0) * sin(mu[0]) + x[0] + x[1] * (sqrt(2.0) * sin(mu[0]) - 1) - 2",
         "- 2 * sqrt(2.0) * cos(mu[0]) + x[1] * (- sqrt(2.0) * cos(mu[0]) + 2) + 2"),  # subdomain 2
        ("- 2 * sqrt(2.0) * cos(mu[0]) + x[0] * (sqrt(2.0) * sin(mu[0]) / 2 - 3 * sqrt(2.0) * cos(mu[0]) / 2 + 2)"
         + "+ x[1] * (- sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2) + 2",
         "- 2 * sqrt(2.0) * sin(mu[0]) + x[0] * (- 3 * sqrt(2.0) * sin(mu[0]) / 2 - sqrt(2.0) * cos(mu[0]) / 2 + 2)"
         + "+ x[1] * (sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2) + 2"),  # subdomain 3
        ("- 2 * sqrt(2.0) * sin(mu[0]) + x[0] * (- sqrt(2.0) * sin(mu[0]) + 2) + 2",
         "2 * sqrt(2.0) * cos(mu[0]) + x[0] * (sqrt(2.0) * cos(mu[0]) - 1) + x[1] - 2"),  # subdomain 4
        ("2 * sqrt(2.0) * sin(mu[0]) + x[0]* (- 3 * sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2 + 2)"
         + "+ x[1] * (- sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2) - 2",
         "- 2 * sqrt(2.0) * cos(mu[0]) + x[0] * (sqrt(2.0) * sin(mu[0]) / 2 + 3 * sqrt(2.0) * cos(mu[0]) / 2 - 2)"
         + "+ x[1] * (sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2) + 2"),  # subdomain 5
        ("2 * sqrt(2.0) * cos(mu[0]) + x[0] * (- sqrt(2.0) * cos(mu[0]) + 2) - 2",
         "2 * sqrt(2.0) * sin(mu[0]) + x[0] * (- sqrt(2.0) * sin(mu[0]) + 1) + x[1] - 2"),  # subdomain 6
        ("- 2 * sqrt(2.0) * sin(mu[0]) + x[0] * (sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2)"
         + "+ x[1] * (3 * sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2 - 2) + 2",
         "2 * sqrt(2.0) * cos(mu[0]) + x[0] * (sqrt(2.0) * sin(mu[0]) / 2 - sqrt(2.0) * cos(mu[0]) / 2)"
         + "+ x[1] * (sqrt(2.0) * sin(mu[0]) / 2 - 3 * sqrt(2.0) * cos(mu[0]) / 2 + 2) - 2"),  # subdomain 7
        ("2 * sqrt(2.0) * cos(mu[0]) + x[0] + x[1] * (- sqrt(2.0) * cos(mu[0]) + 1) - 2",
         "2 * sqrt(2.0) * sin(mu[0]) + x[1] * (- sqrt(2.0) * sin(mu[0]) + 2) - 2")  # subdomain 8
    ]
    shape_parametrization_gradient_expression = [
        compute_shape_parametrization_gradient(expression_on_subdomain)
        for expression_on_subdomain in shape_parametrization_expression]
    # Auxiliary symbolic quantities
    x = MatrixSymbol("x", 2, 1)
    mu = MatrixSymbol("mu", 1, 1)
    # Start checks
    assert len(shape_parametrization_gradient_expression) == 8
    # Check subdomain 1
    assert len(shape_parametrization_gradient_expression[0]) == 2
    assert len(shape_parametrization_gradient_expression[0][X]) == 2
    assert symbolic_equal(
        shape_parametrization_gradient_expression[0][X][X],
        "sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_gradient_expression[0][X][Y],
        "- sqrt(2.0) * sin(mu[0]) / 2 - 3 * sqrt(2.0) * cos(mu[0]) / 2 + 2",
        x, mu)
    assert len(shape_parametrization_gradient_expression[0][Y]) == 2
    assert symbolic_equal(
        shape_parametrization_gradient_expression[0][Y][X],
        "sqrt(2.0) * sin(mu[0]) / 2 - sqrt(2.0) * cos(mu[0]) / 2",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_gradient_expression[0][Y][Y],
        "-3 * sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2 + 2",
        x, mu)
    # Check subdomain 2
    assert len(shape_parametrization_gradient_expression[1]) == 2
    assert len(shape_parametrization_gradient_expression[1][X]) == 2
    assert symbolic_equal(
        shape_parametrization_gradient_expression[1][X][X],
        "1",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_gradient_expression[1][X][Y],
        "sqrt(2.0) * sin(mu[0]) - 1",
        x, mu)
    assert len(shape_parametrization_gradient_expression[1][Y]) == 2
    assert symbolic_equal(
        shape_parametrization_gradient_expression[1][Y][X],
        "0",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_gradient_expression[1][Y][Y],
        "- sqrt(2.0) * cos(mu[0]) + 2",
        x, mu)
    # Check subdomain 3
    assert len(shape_parametrization_gradient_expression[2]) == 2
    assert len(shape_parametrization_gradient_expression[2][X]) == 2
    assert symbolic_equal(
        shape_parametrization_gradient_expression[2][X][X],
        "sqrt(2.0) * sin(mu[0]) / 2 - 3 * sqrt(2.0) * cos(mu[0]) / 2 + 2",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_gradient_expression[2][X][Y],
        "- sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2",
        x, mu)
    assert len(shape_parametrization_gradient_expression[2][Y]) == 2
    assert symbolic_equal(
        shape_parametrization_gradient_expression[2][Y][X],
        "- 3 * sqrt(2.0) * sin(mu[0]) / 2 - sqrt(2.0) * cos(mu[0]) / 2 + 2",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_gradient_expression[2][Y][Y],
        "sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2",
        x, mu)
    # Check subdomain 4
    assert len(shape_parametrization_gradient_expression[3]) == 2
    assert len(shape_parametrization_gradient_expression[3][X]) == 2
    assert symbolic_equal(
        shape_parametrization_gradient_expression[3][X][X],
        "- sqrt(2.0) * sin(mu[0]) + 2",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_gradient_expression[3][X][Y],
        "0",
        x, mu)
    assert len(shape_parametrization_gradient_expression[3][Y]) == 2
    assert symbolic_equal(
        shape_parametrization_gradient_expression[3][Y][X],
        "sqrt(2.0) * cos(mu[0]) - 1",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_gradient_expression[3][Y][Y],
        "1",
        x, mu)
    # Check subdomain 5
    assert len(shape_parametrization_gradient_expression[4]) == 2
    assert len(shape_parametrization_gradient_expression[4][X]) == 2
    assert symbolic_equal(
        shape_parametrization_gradient_expression[4][X][X],
        "- 3 * sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2 + 2",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_gradient_expression[4][X][Y],
        "- sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2",
        x, mu)
    assert len(shape_parametrization_gradient_expression[4][Y]) == 2
    assert symbolic_equal(
        shape_parametrization_gradient_expression[4][Y][X],
        "sqrt(2.0) * sin(mu[0]) / 2 + 3 * sqrt(2.0) * cos(mu[0]) / 2 - 2",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_gradient_expression[4][Y][Y],
        "sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2",
        x, mu)
    # Check subdomain 6
    assert len(shape_parametrization_gradient_expression[5]) == 2
    assert len(shape_parametrization_gradient_expression[5][X]) == 2
    assert symbolic_equal(
        shape_parametrization_gradient_expression[5][X][X],
        "- sqrt(2.0) * cos(mu[0]) + 2",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_gradient_expression[5][X][Y],
        "0",
        x, mu)
    assert len(shape_parametrization_gradient_expression[5][Y]) == 2
    assert symbolic_equal(
        shape_parametrization_gradient_expression[5][Y][X],
        "- sqrt(2.0) * sin(mu[0]) + 1",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_gradient_expression[5][Y][Y],
        "1",
        x, mu)
    # Check subdomain 7
    assert len(shape_parametrization_gradient_expression[6]) == 2
    assert len(shape_parametrization_gradient_expression[6][X]) == 2
    assert symbolic_equal(
        shape_parametrization_gradient_expression[6][X][X],
        "sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_gradient_expression[6][X][Y],
        "3 * sqrt(2.0) * sin(mu[0]) / 2 + sqrt(2.0) * cos(mu[0]) / 2 - 2",
        x, mu)
    assert len(shape_parametrization_gradient_expression[6][Y]) == 2
    assert symbolic_equal(
        shape_parametrization_gradient_expression[6][Y][X],
        "sqrt(2.0) * sin(mu[0]) / 2 - sqrt(2.0) * cos(mu[0]) / 2",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_gradient_expression[6][Y][Y],
        "sqrt(2.0) * sin(mu[0]) / 2 - 3 * sqrt(2.0) * cos(mu[0]) / 2 + 2",
        x, mu)
    # Check subdomain 8
    assert len(shape_parametrization_gradient_expression[7]) == 2
    assert len(shape_parametrization_gradient_expression[7][X]) == 2
    assert symbolic_equal(
        shape_parametrization_gradient_expression[7][X][X],
        "1",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_gradient_expression[7][X][Y],
        "- sqrt(2.0) * cos(mu[0]) + 1",
        x, mu)
    assert len(shape_parametrization_gradient_expression[7][Y]) == 2
    assert symbolic_equal(
        shape_parametrization_gradient_expression[7][Y][X],
        "0",
        x, mu)
    assert symbolic_equal(
        shape_parametrization_gradient_expression[7][Y][Y],
        "- sqrt(2.0) * sin(mu[0]) + 2",
        x, mu)


# Test shape parametrization gradient computation for tutorial 04
def test_compute_shape_parametrization_gradient_graetz():
    shape_parametrization_expression = [
        ("x[0]", "x[1]"),  # subdomain 1
        ("mu[0] * (x[0] - 1) + 1", "x[1]")  # subdomain 2
    ]
    shape_parametrization_gradient_expression = [
        compute_shape_parametrization_gradient(expression_on_subdomain)
        for expression_on_subdomain in shape_parametrization_expression]
    # Auxiliary symbolic quantities
    x = MatrixSymbol("x", 2, 1)
    mu = MatrixSymbol("mu", 1, 1)
    # Start checks
    assert len(shape_parametrization_gradient_expression) == 2
    # Check subdomain 1
    assert len(shape_parametrization_gradient_expression[0]) == 2
    assert len(shape_parametrization_gradient_expression[0][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[0][X][X], "1", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[0][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[0][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[0][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[0][Y][Y], "1", x, mu)
    # Check subdomain 2
    assert len(shape_parametrization_gradient_expression[1]) == 2
    assert len(shape_parametrization_gradient_expression[1][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[1][X][X], "mu[0]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[1][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[1][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[1][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[1][Y][Y], "1", x, mu)


# Test shape parametrization gradient computation for tutorial 12
def test_compute_shape_parametrization_gradient_stokes():
    shape_parametrization_expression = [
        ("mu[4] * x[0] + mu[1] - mu[4]",
         "mu[4] * tan(mu[5]) * x[0] + mu[0] * x[1] + mu[2] - tan(mu[5]) - mu[0]"),  # subdomain 1
        ("mu[4] * x[0] + mu[1] - mu[4]",
         "mu[4] * tan(mu[5]) * x[0] + mu[0] * x[1] + mu[2] - tan(mu[5]) - mu[0]"),  # subdomain 2
        ("mu[1] * x[0]", "mu[3] * x[1] + mu[2] + mu[0] - 2*mu[3]"),  # subdomain 3
        ("mu[1] * x[0]", "mu[3] * x[1] + mu[2] + mu[0] - 2*mu[3]"),  # subdomain 4
        ("mu[1] * x[0]", "mu[0] * x[1] + mu[2] - mu[0]"),  # subdomain 5
        ("mu[1] * x[0]", "mu[0] * x[1] + mu[2] - mu[0]"),  # subdomain 6
        ("mu[1] * x[0]", "mu[2] * x[1]"),  # subdomain 7
        ("mu[1] * x[0]", "mu[2] * x[1]"),  # subdomain 8
    ]
    shape_parametrization_gradient_expression = [
        compute_shape_parametrization_gradient(expression_on_subdomain)
        for expression_on_subdomain in shape_parametrization_expression]
    # Auxiliary symbolic quantities
    x = MatrixSymbol("x", 2, 1)
    mu = MatrixSymbol("mu", 6, 1)
    # Start checks
    assert len(shape_parametrization_gradient_expression) == 8
    # Check subdomain 1
    assert len(shape_parametrization_gradient_expression[0]) == 2
    assert len(shape_parametrization_gradient_expression[0][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[0][X][X], "mu[4]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[0][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[0][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[0][Y][X], "mu[4] * tan(mu[5])", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[0][Y][Y], "mu[0]", x, mu)
    # Check subdomain 2
    assert len(shape_parametrization_gradient_expression[1]) == 2
    assert len(shape_parametrization_gradient_expression[1][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[1][X][X], "mu[4]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[1][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[1][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[1][Y][X], "mu[4] * tan(mu[5])", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[1][Y][Y], "mu[0]", x, mu)
    # Check subdomain 3
    assert len(shape_parametrization_gradient_expression[2]) == 2
    assert len(shape_parametrization_gradient_expression[2][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[2][X][X], "mu[1]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[2][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[2][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[2][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[2][Y][Y], "mu[3]", x, mu)
    # Check subdomain 4
    assert len(shape_parametrization_gradient_expression[3]) == 2
    assert len(shape_parametrization_gradient_expression[3][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[3][X][X], "mu[1]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[3][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[3][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[3][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[3][Y][Y], "mu[3]", x, mu)
    # Check subdomain 5
    assert len(shape_parametrization_gradient_expression[4]) == 2
    assert len(shape_parametrization_gradient_expression[4][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[4][X][X], "mu[1]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[4][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[4][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[4][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[4][Y][Y], "mu[0]", x, mu)
    # Check subdomain 6
    assert len(shape_parametrization_gradient_expression[5]) == 2
    assert len(shape_parametrization_gradient_expression[5][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[5][X][X], "mu[1]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[5][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[5][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[5][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[5][Y][Y], "mu[0]", x, mu)
    # Check subdomain 7
    assert len(shape_parametrization_gradient_expression[6]) == 2
    assert len(shape_parametrization_gradient_expression[6][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[6][X][X], "mu[1]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[6][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[6][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[6][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[6][Y][Y], "mu[2]", x, mu)
    # Check subdomain 8
    assert len(shape_parametrization_gradient_expression[7]) == 2
    assert len(shape_parametrization_gradient_expression[7][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[7][X][X], "mu[1]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[7][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[7][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[7][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[7][Y][Y], "mu[2]", x, mu)


# Test shape parametrization gradient computation for tutorial 17
def test_compute_shape_parametrization_gradient_navier_stokes():
    shape_parametrization_expression = [
        ("x[0]", "x[1]"),  # subdomain 1 bottom
        ("x[0]", "x[1]"),  # subdomain 1 top
        ("x[0]", "0.5 * mu[1] * x[1] - 1.0 * mu[1] + 2.0"),  # subdomain 2 bottom
        ("x[0]", "0.5 * mu[1] * x[1] - 1.0 * mu[1] + 2.0")  # subdomain 2 top
    ]
    shape_parametrization_gradient_expression = [
        compute_shape_parametrization_gradient(expression_on_subdomain)
        for expression_on_subdomain in shape_parametrization_expression]
    # Auxiliary symbolic quantities
    x = MatrixSymbol("x", 2, 1)
    mu = MatrixSymbol("mu", 2, 1)
    # Start checks
    assert len(shape_parametrization_gradient_expression) == 4
    # Check subdomain 1
    assert len(shape_parametrization_gradient_expression[0]) == 2
    assert len(shape_parametrization_gradient_expression[0][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[0][X][X], "1", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[0][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[0][Y]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[0][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[0][Y][Y], "1", x, mu)
    # Check subdomain 2
    assert len(shape_parametrization_gradient_expression[1]) == 2
    assert len(shape_parametrization_gradient_expression[1][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[1][X][X], "1", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[1][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[1][Y]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[1][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[1][Y][Y], "1", x, mu)
    # Check subdomain 3
    assert len(shape_parametrization_gradient_expression[2]) == 2
    assert len(shape_parametrization_gradient_expression[2][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[2][X][X], "1.0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[2][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[2][Y]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[2][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[2][Y][Y], "0.5 * mu[1]", x, mu)
    # Check subdomain 4
    assert len(shape_parametrization_gradient_expression[3]) == 2
    assert len(shape_parametrization_gradient_expression[3][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[3][X][X], "1.0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[3][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[3][Y]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[3][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[3][Y][Y], "0.5 * mu[1]", x, mu)


# Test shape parametrization gradient computation for stokes optimal dirichlet boundary control
def test_compute_shape_parametrization_gradient_stokes_optimal_dirichlet_boundary_control():
    shape_parametrization_expression = [
        ("x[0]", "x[1]"),  # subdomain 1
        ("0.9 - 9.0 * mu[0] + 10.0 * mu[0] * x[0] ", "x[1]"),  # subdomain 2
        ("0.9 - 9.0 * mu[0] + 10.0 * mu[0] * x[0] ", "x[1]"),  # subdomain 3
        ("0.9 - 9.0 * mu[0] + 10.0 * mu[0] * x[0] ", "x[1]"),  # subdomain 4
        ("0.9 - 9.0 * mu[0] + 10.0 * mu[0] * x[0] ", "x[1]"),  # subdomain 5
        ("2.25 * mu[0] + x[0] * (- 1.25 * mu[0] + 1.125) - 0.225 ", "x[1]"),  # subdomain 6
        ("2.0 * mu[0] + x[0] *(- mu[0] + 1.1) + x[1] * (-mu[0] + 0.1) - 0.2 ", "x[1]"),  # subdomain 7
        ("2.25 * mu[0] + x[0] * (- 1.25 * mu[0] + 1.125) - 0.225 ", "x[1]"),  # subdomain 8
        ("mu[0] + x[0] * (- mu[0] + 1.1) + x[1] * (mu[0] - 0.1) - 0.1", "x[1]"),  # subdomain 9
        ("x[0]", "x[1]"),  # subdomain 10
        ("x[0]", "x[1]"),  # subdomain 11
        ("2.25 * mu[0] + x[0] * (- 1.25 * mu[0] + 1.125) - 0.225 ", "x[1]"),  # subdomain 12
        ("2.25 * mu[0] + x[0] * (- 1.25 * mu[0] + 1.125) - 0.225 ", "x[1]")  # subdomain 13
    ]
    shape_parametrization_gradient_expression = [
        compute_shape_parametrization_gradient(expression_on_subdomain)
        for expression_on_subdomain in shape_parametrization_expression]
    # Auxiliary symbolic quantities
    x = MatrixSymbol("x", 2, 1)
    mu = MatrixSymbol("mu", 1, 1)
    # Start checks
    assert len(shape_parametrization_gradient_expression) == 13
    # Check subdomain 1
    assert len(shape_parametrization_gradient_expression[0]) == 2
    assert len(shape_parametrization_gradient_expression[0][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[0][X][X], "1", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[0][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[0][Y]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[0][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[0][Y][Y], "1", x, mu)
    # Check subdomain 2
    assert len(shape_parametrization_gradient_expression[1]) == 2
    assert len(shape_parametrization_gradient_expression[1][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[1][X][X], "10.0 * mu[0]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[1][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[1][Y]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[1][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[1][Y][Y], "1", x, mu)
    # Check subdomain 3
    assert len(shape_parametrization_gradient_expression[2]) == 2
    assert len(shape_parametrization_gradient_expression[2][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[2][X][X], "10.0 * mu[0]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[2][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[2][Y]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[2][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[2][Y][Y], "1", x, mu)
    # Check subdomain 4
    assert len(shape_parametrization_gradient_expression[3]) == 2
    assert len(shape_parametrization_gradient_expression[3][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[3][X][X], "10.0 * mu[0]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[3][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[3][Y]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[3][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[3][Y][Y], "1", x, mu)
    # Check subdomain 5
    assert len(shape_parametrization_gradient_expression[4]) == 2
    assert len(shape_parametrization_gradient_expression[4][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[4][X][X], "10.0 * mu[0]", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[4][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[4][Y]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[4][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[4][Y][Y], "1", x, mu)
    # Check subdomain 6
    assert len(shape_parametrization_gradient_expression[5]) == 2
    assert len(shape_parametrization_gradient_expression[5][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[5][X][X], "- 1.25 * mu[0] + 1.125", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[5][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[5][Y]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[5][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[5][Y][Y], "1", x, mu)
    # Check subdomain 7
    assert len(shape_parametrization_gradient_expression[6]) == 2
    assert len(shape_parametrization_gradient_expression[6][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[6][X][X], "- mu[0] + 1.1", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[6][X][Y], "- mu[0] + 0.1", x, mu)
    assert len(shape_parametrization_gradient_expression[6][Y]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[6][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[6][Y][Y], "1", x, mu)
    # Check subdomain 8
    assert len(shape_parametrization_gradient_expression[7]) == 2
    assert len(shape_parametrization_gradient_expression[7][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[7][X][X], "-1.25 * mu[0] + 1.125", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[7][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[7][Y]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[7][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[7][Y][Y], "1", x, mu)
    # Check subdomain 9
    assert len(shape_parametrization_gradient_expression[8]) == 2
    assert len(shape_parametrization_gradient_expression[8][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[8][X][X], "- mu[0] + 1.1", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[8][X][Y], "mu[0] - 0.1", x, mu)
    assert len(shape_parametrization_gradient_expression[8][Y]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[8][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[8][Y][Y], "1", x, mu)
    # Check subdomain 10
    assert len(shape_parametrization_gradient_expression[9]) == 2
    assert len(shape_parametrization_gradient_expression[9][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[9][X][X], "1", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[9][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[9][Y]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[9][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[9][Y][Y], "1", x, mu)
    # Check subdomain 11
    assert len(shape_parametrization_gradient_expression[10]) == 2
    assert len(shape_parametrization_gradient_expression[10][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[10][X][X], "1", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[10][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[10][Y]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[10][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[10][Y][Y], "1", x, mu)
    # Check subdomain 12
    assert len(shape_parametrization_gradient_expression[11]) == 2
    assert len(shape_parametrization_gradient_expression[11][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[11][X][X], "- 1.25 * mu[0] + 1.125", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[11][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[11][Y]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[11][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[11][Y][Y], "1", x, mu)
    # Check subdomain 13
    assert len(shape_parametrization_gradient_expression[12]) == 2
    assert len(shape_parametrization_gradient_expression[12][X]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[12][X][X], "- 1.25 * mu[0] + 1.125", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[12][X][Y], "0", x, mu)
    assert len(shape_parametrization_gradient_expression[12][Y]) == 2
    assert symbolic_equal(shape_parametrization_gradient_expression[12][Y][X], "0", x, mu)
    assert symbolic_equal(shape_parametrization_gradient_expression[12][Y][Y], "1", x, mu)
