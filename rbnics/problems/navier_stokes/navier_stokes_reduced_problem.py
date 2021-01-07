# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base import NonlinearReducedProblem
from rbnics.backends import product, sum


def NavierStokesReducedProblem(StokesReducedProblem_DerivedClass):

    NavierStokesReducedProblem_Base = NonlinearReducedProblem(StokesReducedProblem_DerivedClass)

    class NavierStokesReducedProblem_Class(NavierStokesReducedProblem_Base):

        class ProblemSolver(NavierStokesReducedProblem_Base.ProblemSolver):
            def residual_eval(self, solution):
                problem = self.problem
                N = self.N
                assembled_operator = dict()
                for term in ("a", "b", "bt", "c", "f", "g"):
                    assert problem.terms_order[term] in (1, 2)
                    if problem.terms_order[term] == 2:
                        assembled_operator[term] = sum(product(
                            problem.compute_theta(term), problem.operator[term][:N, :N]))
                    elif problem.terms_order[term] == 1:
                        assembled_operator[term] = sum(product(
                            problem.compute_theta(term), problem.operator[term][:N]))
                    else:
                        raise ValueError("Invalid value for order of term " + term)
                return ((assembled_operator["a"] + assembled_operator["b"] + assembled_operator["bt"]) * solution
                        + assembled_operator["c"]
                        - assembled_operator["f"] - assembled_operator["g"])

            def jacobian_eval(self, solution):
                problem = self.problem
                N = self.N
                assembled_operator = dict()
                for term in ("a", "b", "bt", "dc"):
                    assert problem.terms_order[term] == 2
                    assembled_operator[term] = sum(product(
                        problem.compute_theta(term), problem.operator[term][:N, :N]))
                return (assembled_operator["a"] + assembled_operator["b"] + assembled_operator["bt"]
                        + assembled_operator["dc"])

    # return value (a class) for the decorator
    return NavierStokesReducedProblem_Class
