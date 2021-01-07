# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base import NonlinearTimeDependentReducedProblem
from rbnics.backends import product, sum


def NonlinearParabolicReducedProblem(NonlinearEllipticReducedProblem_DerivedClass):

    NonlinearParabolicReducedProblem_Base = NonlinearTimeDependentReducedProblem(
        NonlinearEllipticReducedProblem_DerivedClass)

    class NonlinearParabolicReducedProblem_Class(NonlinearParabolicReducedProblem_Base):

        class ProblemSolver(NonlinearParabolicReducedProblem_Base.ProblemSolver):
            def residual_eval(self, t, solution, solution_dot):
                problem = self.problem
                N = self.N
                assembled_operator = dict()
                assembled_operator["m"] = sum(product(problem.compute_theta("m"), problem.operator["m"][:N, :N]))
                assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"][:N, :N]))
                assembled_operator["c"] = sum(product(problem.compute_theta("c"), problem.operator["c"][:N]))
                assembled_operator["f"] = sum(product(problem.compute_theta("f"), problem.operator["f"][:N]))
                return (assembled_operator["m"] * solution_dot
                        + assembled_operator["a"] * solution
                        + assembled_operator["c"]
                        - assembled_operator["f"])

            def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
                problem = self.problem
                N = self.N
                assembled_operator = dict()
                assembled_operator["m"] = sum(product(problem.compute_theta("m"), problem.operator["m"][:N, :N]))
                assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"][:N, :N]))
                assembled_operator["dc"] = sum(product(problem.compute_theta("dc"), problem.operator["dc"][:N, :N]))
                return (assembled_operator["m"] * solution_dot_coefficient
                        + assembled_operator["a"]
                        + assembled_operator["dc"])

    # return value (a class) for the decorator
    return NonlinearParabolicReducedProblem_Class
