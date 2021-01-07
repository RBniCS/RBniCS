# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base import LinearProblem, ParametrizedDifferentialProblem
from rbnics.backends import product, sum, transpose

GeostrophicOptimalControlProblem_Base = LinearProblem(ParametrizedDifferentialProblem)


class GeostrophicOptimalControlProblem(GeostrophicOptimalControlProblem_Base):

    """
    The problem to be solved is
            min {J(y_psi, u) = 1/2 m(y_psi - y_d, y_psi - y_d) + 1/2 n(u, u)}
            (y_psi, y_q) in YxY,  y_d in Y, u in U
            s.t.
            a((y_psi, y_q), (z_psi, z_q)) = c(u, z_psi) + <f, z_q>    for all (z_psi, z_q) in YxY

    This class will solve the following optimality conditions:
            m(y_psi, z_q) + a*((p_psi, p_q), (z_psi, z_q)) = <g, z_q>     for all (z_psi, z_q) in YxY
            n(u, v) - c*(v, ppsi)                          = 0            for all v in U
            a((y_psi, y_q), (q_psi, q_q))- c(u, q_psi)     = <f, q_q>     for all (q_psi, q_q) in YxY

    and compute the cost functional
            J(y_psi, u) = 1/2 m(y_psi, y_psi) + 1/2 n(u, u) - <g, y_psi> + 1/2 h

    where
            a*(., .) is the adjoint of a
            c*(., .) is the adjoint of c
            <g, y_psi> = m(y_d, y_psi)
            h = m(y_d, y_d)

    """

    def __init__(self, V, **kwargs):
        # Call to parent
        GeostrophicOptimalControlProblem_Base.__init__(self, V, **kwargs)

        # Form names for saddle point problems
        self.terms = [
            "a", "a*", "c", "c*", "m", "n", "f", "g", "h"
        ]
        self.terms_order = {
            "a": 2, "a*": 2,
            "c": 2, "c*": 2,
            "m": 2, "n": 2,
            "f": 1, "g": 1,
            "h": 0
        }
        self.components = ["ypsi", "yq", "u", "ppsi", "pq"]

    class ProblemSolver(GeostrophicOptimalControlProblem_Base.ProblemSolver):
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

    def _compute_output(self):
        assembled_operator = dict()
        for term in ("m", "n", "g", "h"):
            assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term]))
        self._output = (0.5 * (transpose(self._solution) * assembled_operator["m"] * self._solution)
                        + 0.5 * (transpose(self._solution) * assembled_operator["n"] * self._solution)
                        - transpose(assembled_operator["g"]) * self._solution
                        + 0.5 * assembled_operator["h"])
