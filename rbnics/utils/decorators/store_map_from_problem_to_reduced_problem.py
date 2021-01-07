# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.cache import Cache
from rbnics.utils.decorators.preserve_class_name import PreserveClassName
from rbnics.utils.jupyter import is_jupyter


def StoreMapFromProblemToReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    @PreserveClassName
    class StoreMapFromProblemToReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):

        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)

            # Populate problem to reduced problem map
            add_to_map_from_problem_to_reduced_problem(truth_problem, self)

    # return value (a class) for the decorator
    return StoreMapFromProblemToReducedProblem_Class


def add_to_map_from_problem_to_reduced_problem(problem, reduced_problem):
    if problem not in _problem_to_reduced_problem_map or is_jupyter():
        if hasattr(type(problem), "__is_exact__"):
            problem = problem.__decorated_problem__
        _problem_to_reduced_problem_map[problem] = reduced_problem
    else:
        assert _problem_to_reduced_problem_map[problem] is reduced_problem


def get_reduced_problem_from_problem(problem):
    assert problem in _problem_to_reduced_problem_map
    return _problem_to_reduced_problem_map[problem]


_problem_to_reduced_problem_map = Cache()
