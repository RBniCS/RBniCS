# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base import LinearReducedProblem
from rbnics.backends import product, sum, transpose


def EllipticReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    EllipticReducedProblem_Base = LinearReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass)

    # Base class containing the interface of a projection based ROM
    # for elliptic problems.
    class EllipticReducedProblem_Class(EllipticReducedProblem_Base):

        class ProblemSolver(EllipticReducedProblem_Base.ProblemSolver):
            def matrix_eval(self):
                problem = self.problem
                N = self.N
                return sum(product(problem.compute_theta("a"), problem.operator["a"][:N, :N]))

            def vector_eval(self):
                problem = self.problem
                N = self.N
                return sum(product(problem.compute_theta("f"), problem.operator["f"][:N]))

        # Perform an online evaluation of the output
        def _compute_output(self, N):
            self._output = transpose(self._solution) * sum(product(self.compute_theta("s"), self.operator["s"][:N]))

    # return value (a class) for the decorator
    return EllipticReducedProblem_Class
