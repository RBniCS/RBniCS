# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base import NonlinearReducedProblem
from rbnics.backends import product, sum, transpose


def NonlinearEllipticReducedProblem(EllipticReducedProblem_DerivedClass):

    NonlinearEllipticReducedProblem_Base = NonlinearReducedProblem(EllipticReducedProblem_DerivedClass)

    class NonlinearEllipticReducedProblem_Class(NonlinearEllipticReducedProblem_Base):

        class ProblemSolver(NonlinearEllipticReducedProblem_Base.ProblemSolver):
            def residual_eval(self, solution):
                problem = self.problem
                N = self.N
                assembled_operator = dict()
                assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"][:N, :N]))
                assembled_operator["c"] = sum(product(problem.compute_theta("c"), problem.operator["c"][:N]))
                assembled_operator["f"] = sum(product(problem.compute_theta("f"), problem.operator["f"][:N]))
                return assembled_operator["a"] * solution + assembled_operator["c"] - assembled_operator["f"]

            def jacobian_eval(self, solution):
                problem = self.problem
                N = self.N
                assembled_operator = dict()
                assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"][:N, :N]))
                assembled_operator["dc"] = sum(product(problem.compute_theta("dc"), problem.operator["dc"][:N, :N]))
                return assembled_operator["a"] + assembled_operator["dc"]

        # Perform an online evaluation of the output
        def _compute_output(self, N):
            self._output = transpose(self._solution) * sum(product(self.compute_theta("s"), self.operator["s"][:N]))

    # return value (a class) for the decorator
    return NonlinearEllipticReducedProblem_Class
