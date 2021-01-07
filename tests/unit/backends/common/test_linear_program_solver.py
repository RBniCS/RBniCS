# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numpy import isclose
from rbnics.backends.common.linear_program_solver import LinearProgramSolver, Matrix, Vector

"""
Solve
    min    0.5 x + y
    s.t.   x + y >= 1
         - x + y >= -0.5
           0 <= x <= 1
           0 <= y <= 1
The optimal solution is
    x = 0.75, y = 0.25
with cost
    0.625
"""


def test_linear_program_solver():
    c = Vector(2)
    A = Matrix(2, 2)
    b = Vector(2)
    bounds = [None] * 2

    c[0], c[1] = 0.5, 1.
    A[0, 0], A[0, 1] = 1., 1.
    A[1, 0], A[1, 1] = -1., 1.
    b[0], b[1] = 1., -0.5
    bounds[0] = (0., 1.)
    bounds[1] = (0., 1.)

    solver = LinearProgramSolver(c, A, b, bounds)
    optimal_cost = solver.solve()
    assert isclose(optimal_cost, 0.625)
