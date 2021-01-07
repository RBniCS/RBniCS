# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base import LinearProblem, ParametrizedDifferentialProblem
from rbnics.backends import product, sum, transpose

EllipticOptimalControlProblem_Base = LinearProblem(ParametrizedDifferentialProblem)


# Base class containing the definition of saddle point problems
class EllipticOptimalControlProblem(EllipticOptimalControlProblem_Base):
    """
    The problem to be solved is
        min {J(y, u) = 1/2 m(y - y_d, y - y_d) + 1/2 n(u, u)}
        y in Y, u in U
        s.t.
        a(y, q) = c(u, q) + <f, q>      for all q in Q = Y

    This class will solve the following optimality conditions:
        m(y, z)           + a*(p, z) = <g, z>     for all z in Y
                  n(u, v) - c*(p, v) = 0          for all v in U
        a(y, q) - c(u, q)            = <f, q>     for all q in Q

    and compute the cost functional
        J(y, u) = 1/2 m(y, y) + 1/2 n(u, u) - <g, y> + 1/2 h

    where
        a*(., .) is the adjoint of a
        c*(., .) is the adjoint of c
        <g, y> = m(y_d, y)
        h = m(y_d, y_d)
    """

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call to parent
        EllipticOptimalControlProblem_Base.__init__(self, V, **kwargs)

        # Form names for saddle point problems
        self.terms = ["a", "a*", "c", "c*", "m", "n", "f", "g", "h"]
        self.terms_order = {
            "a": 2, "a*": 2,
            "c": 2, "c*": 2,
            "m": 2, "n": 2,
            "f": 1, "g": 1,
            "h": 0
        }
        self.components = ["y", "u", "p"]

    class ProblemSolver(EllipticOptimalControlProblem_Base.ProblemSolver):
        def matrix_eval(self):
            problem = self.problem
            assembled_operator = dict()
            for term in ("a", "a*", "c", "c*", "m", "n"):
                assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term]))
            return (assembled_operator["m"] + assembled_operator["a*"]
                    + assembled_operator["n"] - assembled_operator["c*"]
                    + assembled_operator["a"] - assembled_operator["c"])

        def vector_eval(self):
            problem = self.problem
            assembled_operator = dict()
            for term in ("f", "g"):
                assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term]))
            return (assembled_operator["g"]
                    + assembled_operator["f"])

    # Perform a truth evaluation of the cost functional
    def _compute_output(self):
        assembled_operator = dict()
        for term in ("g", "h", "m", "n"):
            assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term]))
        self._output = (
            0.5 * (transpose(self._solution) * assembled_operator["m"] * self._solution)
            + 0.5 * (transpose(self._solution) * assembled_operator["n"] * self._solution)
            - transpose(assembled_operator["g"]) * self._solution
            + 0.5 * assembled_operator["h"]
        )
