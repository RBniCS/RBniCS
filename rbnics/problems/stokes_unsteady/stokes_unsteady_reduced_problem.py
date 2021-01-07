# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base import LinearTimeDependentReducedProblem
from rbnics.backends import product, sum


def AbstractCFDUnsteadyReducedProblem(AbstractCFDUnsteadyReducedProblem_Base):
    return AbstractCFDUnsteadyReducedProblem_Base


def StokesUnsteadyReducedProblem(StokesReducedProblem_DerivedClass):

    StokesUnsteadyReducedProblem_Base = AbstractCFDUnsteadyReducedProblem(
        LinearTimeDependentReducedProblem(StokesReducedProblem_DerivedClass))

    class StokesUnsteadyReducedProblem_Class(StokesUnsteadyReducedProblem_Base):

        class ProblemSolver(StokesUnsteadyReducedProblem_Base.ProblemSolver):
            def residual_eval(self, t, solution, solution_dot):
                problem = self.problem
                N = self.N
                assembled_operator = dict()
                for term in ("m", "a", "b", "bt", "f", "g"):
                    assert problem.terms_order[term] in (1, 2)
                    if problem.terms_order[term] == 2:
                        assembled_operator[term] = sum(product(
                            problem.compute_theta(term), problem.operator[term][:N, :N]))
                    elif problem.terms_order[term] == 1:
                        assembled_operator[term] = sum(product(
                            problem.compute_theta(term), problem.operator[term][:N]))
                    else:
                        raise ValueError("Invalid value for order of term " + term)
                return (assembled_operator["m"] * solution_dot
                        + (assembled_operator["a"] + assembled_operator["b"] + assembled_operator["bt"]) * solution
                        - assembled_operator["f"] - assembled_operator["g"])

            def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
                problem = self.problem
                N = self.N
                assembled_operator = dict()
                for term in ("m", "a", "b", "bt"):
                    assembled_operator[term] = sum(product(
                        problem.compute_theta(term), problem.operator[term][:N, :N]))
                return (assembled_operator["m"] * solution_dot_coefficient
                        + assembled_operator["a"] + assembled_operator["b"] + assembled_operator["bt"])

    # return value (a class) for the decorator
    return StokesUnsteadyReducedProblem_Class
