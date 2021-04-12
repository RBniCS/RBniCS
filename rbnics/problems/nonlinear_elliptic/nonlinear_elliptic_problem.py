# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base import NonlinearProblem
from rbnics.problems.elliptic import EllipticProblem
from rbnics.backends import product, sum, transpose

NonlinearEllipticProblem_Base = NonlinearProblem(EllipticProblem)


class NonlinearEllipticProblem(NonlinearEllipticProblem_Base):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call to parent
        NonlinearEllipticProblem_Base.__init__(self, V, **kwargs)

        # Form names for nonlinear problems
        self.terms = ["a", "c", "dc", "f", "s"]
        self.terms_order = {"a": 2, "c": 1, "dc": 2, "f": 1, "s": 1}

    class ProblemSolver(NonlinearEllipticProblem_Base.ProblemSolver):
        def residual_eval(self, solution):
            problem = self.problem
            assembled_operator = dict()
            assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"]))
            assembled_operator["c"] = sum(product(problem.compute_theta("c"), problem.operator["c"]))
            assembled_operator["f"] = sum(product(problem.compute_theta("f"), problem.operator["f"]))
            return assembled_operator["a"] * solution + assembled_operator["c"] - assembled_operator["f"]

        def jacobian_eval(self, solution):
            problem = self.problem
            assembled_operator = dict()
            assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"]))
            assembled_operator["dc"] = sum(product(problem.compute_theta("dc"), problem.operator["dc"]))
            return assembled_operator["a"] + assembled_operator["dc"]

    # Perform a truth evaluation of the output
    def _compute_output(self):
        self._output = transpose(self._solution) * sum(product(self.compute_theta("s"), self.operator["s"]))
