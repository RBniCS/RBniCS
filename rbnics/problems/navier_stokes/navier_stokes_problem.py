# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base import NonlinearProblem
from rbnics.problems.stokes import StokesProblem
from rbnics.backends import product, sum

NavierStokesProblem_Base = NonlinearProblem(StokesProblem)


class NavierStokesProblem(NavierStokesProblem_Base):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call to parent
        NavierStokesProblem_Base.__init__(self, V, **kwargs)

        # Form names for Navier-Stokes problems
        self.terms.extend([
            "c", "dc"
        ])
        self.terms_order.update({
            "c": 1, "dc": 2
        })

    class ProblemSolver(NavierStokesProblem_Base.ProblemSolver):
        def residual_eval(self, solution):
            problem = self.problem
            assembled_operator = dict()
            for term in ("a", "b", "bt", "c", "f", "g"):
                assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term]))
            return ((assembled_operator["a"] + assembled_operator["b"] + assembled_operator["bt"]) * solution
                    + assembled_operator["c"]
                    - assembled_operator["f"] - assembled_operator["g"])

        def jacobian_eval(self, solution):
            problem = self.problem
            assembled_operator = dict()
            for term in ("a", "b", "bt", "dc"):
                assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term]))
            return (assembled_operator["a"] + assembled_operator["b"] + assembled_operator["bt"]
                    + assembled_operator["dc"])
