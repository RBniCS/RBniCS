# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends import product, sum
from rbnics.problems.base import LinearReducedProblem


def GeostrophicReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    GeostrophicReducedProblem_Base = LinearReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass)

    class GeostrophicReducedProblem_Class(GeostrophicReducedProblem_Base):

        class ProblemSolver(GeostrophicReducedProblem_Base.ProblemSolver):
            def matrix_eval(self):
                problem = self.problem
                N = self.N
                return sum(product(problem.compute_theta("a"), problem.operator["a"][:N, :N]))

            def vector_eval(self):
                problem = self.problem
                N = self.N
                return sum(product(problem.compute_theta("f"), problem.operator["f"][:N]))

    # return value (a class) for the decorator
    return GeostrophicReducedProblem_Class
