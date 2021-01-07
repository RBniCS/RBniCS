# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base import LinearProblem, ParametrizedDifferentialProblem
from rbnics.backends import product, sum

GeostrophicProblem_Base = LinearProblem(ParametrizedDifferentialProblem)


class GeostrophicProblem(GeostrophicProblem_Base):
    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call to parent
        GeostrophicProblem_Base.__init__(self, V, **kwargs)

        # Form names for geostrophic problems
        self.terms = ["a", "f"]
        self.terms_order = {"a": 2, "f": 1}
        self.components = ["psi", "q"]

    # Perform a truth solve
    class ProblemSolver(GeostrophicProblem_Base.ProblemSolver):
        def matrix_eval(self):
            problem = self.problem
            return sum(product(problem.compute_theta("a"), problem.operator["a"]))

        def vector_eval(self):
            problem = self.problem
            return sum(product(problem.compute_theta("f"), problem.operator["f"]))
