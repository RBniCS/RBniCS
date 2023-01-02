# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends import product, sum, transpose
from rbnics.problems.base import LinearTimeDependentProblem


def AbstractParabolicProblem(EllipticProblem_DerivedClass):
    AbstractParabolicProblem_Base = LinearTimeDependentProblem(EllipticProblem_DerivedClass)

    class AbstractParabolicProblem_Class(AbstractParabolicProblem_Base):

        # Default initialization of members
        def __init__(self, V, **kwargs):
            # Call to parent
            AbstractParabolicProblem_Base.__init__(self, V, **kwargs)

            # Form names for parabolic problems
            self.terms.append("m")
            self.terms_order.update({"m": 2})

        class ProblemSolver(AbstractParabolicProblem_Base.ProblemSolver):
            def residual_eval(self, t, solution, solution_dot):
                problem = self.problem
                assembled_operator = dict()
                assembled_operator["m"] = sum(product(problem.compute_theta("m"), problem.operator["m"]))
                assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"]))
                assembled_operator["f"] = sum(product(problem.compute_theta("f"), problem.operator["f"]))
                return (assembled_operator["m"] * solution_dot
                        + assembled_operator["a"] * solution
                        - assembled_operator["f"])

            def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
                problem = self.problem
                assembled_operator = dict()
                assembled_operator["m"] = sum(product(problem.compute_theta("m"), problem.operator["m"]))
                assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"]))
                return (assembled_operator["m"] * solution_dot_coefficient
                        + assembled_operator["a"])

        # Perform a truth evaluation of the output
        def _compute_output(self):
            self._output_over_time.clear()
            for sol in self._solution_over_time:
                self._output_over_time.extend(
                    [transpose(sol) * sum(product(self.compute_theta("s"), self.operator["s"]))])
            self._output = self._output_over_time[-1]

    # return value (a class) for the decorator
    return AbstractParabolicProblem_Class
