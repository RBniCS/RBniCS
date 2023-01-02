# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.cache import Cache
from rbnics.utils.decorators.preserve_class_name import PreserveClassName


def StoreMapFromSolutionDotToProblem(ParametrizedDifferentialProblem_DerivedClass):

    @PreserveClassName
    class StoreMapFromSolutionDotToProblem_Class(ParametrizedDifferentialProblem_DerivedClass):

        def __init__(self, V, **kwargs):
            # Call the parent initialization
            ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)

            # Populate solution_dot to problem map
            add_to_map_from_solution_dot_to_problem(self._solution_dot, self)

    # return value (a class) for the decorator
    return StoreMapFromSolutionDotToProblem_Class


def add_to_map_from_solution_dot_to_problem(solution_dot, problem):
    if solution_dot not in _solution_dot_to_problem_map:
        _solution_dot_to_problem_map[solution_dot] = problem
    else:
        assert problem is _solution_dot_to_problem_map[solution_dot]


def get_problem_from_solution_dot(solution_dot):
    assert solution_dot in _solution_dot_to_problem_map
    return _solution_dot_to_problem_map[solution_dot]


_solution_dot_to_problem_map = Cache()
