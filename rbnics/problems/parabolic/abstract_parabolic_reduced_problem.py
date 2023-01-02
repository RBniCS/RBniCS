# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base import LinearTimeDependentReducedProblem
from rbnics.backends import product, sum, transpose


def AbstractParabolicReducedProblem(EllipticReducedProblem_DerivedClass):

    AbstractParabolicReducedProblem_Base = LinearTimeDependentReducedProblem(EllipticReducedProblem_DerivedClass)

    class AbstractParabolicReducedProblem_Class(AbstractParabolicReducedProblem_Base):

        class ProblemSolver(AbstractParabolicReducedProblem_Base.ProblemSolver):
            def residual_eval(self, t, solution, solution_dot):
                problem = self.problem
                N = self.N
                assembled_operator = dict()
                assembled_operator["m"] = sum(product(problem.compute_theta("m"), problem.operator["m"][:N, :N]))
                assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"][:N, :N]))
                assembled_operator["f"] = sum(product(problem.compute_theta("f"), problem.operator["f"][:N]))
                return (assembled_operator["m"] * solution_dot
                        + assembled_operator["a"] * solution
                        - assembled_operator["f"])

            def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
                problem = self.problem
                N = self.N
                assembled_operator = dict()
                assembled_operator["m"] = sum(product(problem.compute_theta("m"), problem.operator["m"][:N, :N]))
                assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"][:N, :N]))
                return (assembled_operator["m"] * solution_dot_coefficient
                        + assembled_operator["a"])

        # Perform an online evaluation of the output
        def _compute_output(self, N):
            self._output_over_time.clear()
            for sol in self._solution_over_time:
                self._output_over_time.extend(
                    [transpose(sol) * sum(product(self.compute_theta("s"), self.operator["s"][:N]))])
            self._output = self._output_over_time[-1]

    # return value (a class) for the decorator
    return AbstractParabolicReducedProblem_Class
