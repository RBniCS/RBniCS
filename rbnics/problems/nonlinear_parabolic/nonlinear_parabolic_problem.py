# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base import NonlinearTimeDependentProblem
from rbnics.problems.nonlinear_elliptic import NonlinearEllipticProblem
from rbnics.backends import product, sum

NonlinearParabolicProblem_Base = NonlinearTimeDependentProblem(NonlinearEllipticProblem)


class NonlinearParabolicProblem(NonlinearParabolicProblem_Base):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call to parent
        NonlinearParabolicProblem_Base.__init__(self, V, **kwargs)

        # Form names for parabolic problems
        self.terms.append("m")
        self.terms_order.update({"m": 2})

    class ProblemSolver(NonlinearParabolicProblem_Base.ProblemSolver):
        def residual_eval(self, t, solution, solution_dot):
            problem = self.problem
            assembled_operator = dict()
            assembled_operator["m"] = sum(product(problem.compute_theta("m"), problem.operator["m"]))
            assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"]))
            assembled_operator["c"] = sum(product(problem.compute_theta("c"), problem.operator["c"]))
            assembled_operator["f"] = sum(product(problem.compute_theta("f"), problem.operator["f"]))
            return (assembled_operator["m"] * solution_dot
                    + assembled_operator["a"] * solution
                    + assembled_operator["c"]
                    - assembled_operator["f"])

        def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
            problem = self.problem
            assembled_operator = dict()
            assembled_operator["m"] = sum(product(problem.compute_theta("m"), problem.operator["m"]))
            assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"]))
            assembled_operator["dc"] = sum(product(problem.compute_theta("dc"), problem.operator["dc"]))
            return (assembled_operator["m"] * solution_dot_coefficient
                    + assembled_operator["a"]
                    + assembled_operator["dc"])
