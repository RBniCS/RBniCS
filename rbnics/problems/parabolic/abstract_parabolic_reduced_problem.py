# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base import LinearTimeDependentReducedProblem
from rbnics.backends import product, sum


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

    # return value (a class) for the decorator
    return AbstractParabolicReducedProblem_Class
