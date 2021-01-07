# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.cache import Cache
from rbnics.utils.decorators.preserve_class_name import PreserveClassName


def StoreMapFromSolutionToProblem(ParametrizedDifferentialProblem_DerivedClass):

    @PreserveClassName
    class StoreMapFromSolutionToProblem_Class(ParametrizedDifferentialProblem_DerivedClass):

        def __init__(self, V, **kwargs):
            # Call the parent initialization
            ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)

            # Populate solution to problem map
            add_to_map_from_solution_to_problem(self._solution, self)

    # return value (a class) for the decorator
    return StoreMapFromSolutionToProblem_Class


def add_to_map_from_solution_to_problem(solution, problem):
    if solution not in _solution_to_problem_map:
        _solution_to_problem_map[solution] = problem
    else:
        assert problem is _solution_to_problem_map[solution]


def get_problem_from_solution(solution):
    assert solution in _solution_to_problem_map
    return _solution_to_problem_map[solution]


_solution_to_problem_map = Cache()
