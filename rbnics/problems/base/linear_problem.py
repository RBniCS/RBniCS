# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends import LinearProblemWrapper, LinearSolver
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(None)
def LinearProblem(ParametrizedDifferentialProblem_DerivedClass):

    @PreserveClassName
    class LinearProblem_Class(ParametrizedDifferentialProblem_DerivedClass):

        # Default initialization of members
        def __init__(self, V, **kwargs):
            # Call to parent
            ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)

            # Nonlinear solver parameters
            self._linear_solver_parameters = dict()

        class ProblemSolver(ParametrizedDifferentialProblem_DerivedClass.ProblemSolver, LinearProblemWrapper):
            def solve(self):
                problem = self.problem
                solver = LinearSolver(self, problem._solution)
                solver.set_parameters(problem._linear_solver_parameters)
                solver.solve()

    # return value (a class) for the decorator
    return LinearProblem_Class
