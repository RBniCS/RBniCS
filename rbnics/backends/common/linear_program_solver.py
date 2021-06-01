# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import cvxopt
from numbers import Number
from numpy import eye, hstack, isclose, matrix as numpy_matrix, ndarray as numpy_vector, vstack, zeros
from rbnics.backends.abstract import LinearProgramSolver as AbstractLinearProgramSolver
from rbnics.utils.decorators import BackendFor, list_of, tuple_of


# Helper classes for linear pogram
def Matrix(m, n):
    return numpy_matrix(zeros((m, n)))


def Vector(n):
    return zeros((n, ))


class Error(RuntimeError):
    pass


# Linear program solver
@BackendFor("common", inputs=(numpy_vector, numpy_matrix, numpy_vector, list_of(tuple_of(Number))))
class LinearProgramSolver(AbstractLinearProgramSolver):
    def __init__(self, cost, inequality_constraints_matrix, inequality_constraints_vector, bounds):
        self.Q = len(cost)
        # Store cost
        self.cost = cvxopt.matrix(cost)
        # Store inequality constraints matrix, also including a 2*Q x 2*Q submatrix for bound constraints
        self.inequality_constraints_matrix = cvxopt.matrix(vstack((- inequality_constraints_matrix, - eye(self.Q),
                                                                   eye(self.Q))))
        # Store inequality constraints vector, also including 2*Q rows for bound constraints
        assert len(bounds) == self.Q
        bounds_lower = zeros(self.Q)
        bounds_upper = zeros(self.Q)
        for (q, bounds_q) in enumerate(bounds):
            assert bounds_q[0] <= bounds_q[1] or isclose(bounds_q[0], bounds_q[1])
            if bounds_q[0] <= bounds_q[1]:
                bounds_lower[q] = bounds_q[0]
                bounds_upper[q] = bounds_q[1]
            else:
                bounds_lower[q] = bounds_q[1]
                bounds_upper[q] = bounds_q[0]
        self.inequality_constraints_vector = cvxopt.matrix(hstack((- inequality_constraints_vector,
                                                                   bounds_lower, bounds_upper)))

    def solve(self):
        result = cvxopt.solvers.lp(self.cost, self.inequality_constraints_matrix, self.inequality_constraints_vector,
                                   solver="glpk", options={"glpk": {"msg_lev": "GLP_MSG_OFF"}})
        if result["status"] != "optimal":
            raise Error("Linear program solver reports convergence failure with reason", result["status"])
        else:
            return result["primal objective"]
