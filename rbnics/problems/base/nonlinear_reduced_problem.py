# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends import NonlinearProblemWrapper, NonlinearSolver
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(None)
def NonlinearReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    @PreserveClassName
    class NonlinearReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):

        # Default initialization of members.
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)

            # Nonlinear solver parameters
            self._nonlinear_solver_parameters = dict()

        class ProblemSolver(ParametrizedReducedDifferentialProblem_DerivedClass.ProblemSolver, NonlinearProblemWrapper):
            def solve(self):
                problem = self.problem
                solver = NonlinearSolver(self, problem._solution)
                solver.set_parameters(problem._nonlinear_solver_parameters)
                solver.solve()

    # return value (a class) for the decorator
    return NonlinearReducedProblem_Class
