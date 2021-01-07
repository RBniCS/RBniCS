# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.cache import Cache
from rbnics.utils.decorators.preserve_class_name import PreserveClassName
from rbnics.utils.jupyter import is_jupyter


def StoreMapFromProblemNameToProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    @PreserveClassName
    class StoreMapFromProblemNameToProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):

        def __init__(self, V, **kwargs):
            # Call the parent initialization
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)

            # Populate problem name to problem map
            add_to_map_from_problem_name_to_problem(self.name(), self)

    # return value (a class) for the decorator
    return StoreMapFromProblemNameToProblem_Class


def add_to_map_from_problem_name_to_problem(problem_name, problem):
    if hasattr(type(problem), "__is_exact__"):
        assert type(problem).__is_exact__ is True
        problem_name = problem.__decorated_problem__.name()
        assert problem_name in _problem_name_to_problem_map
    else:
        if problem_name not in _problem_name_to_problem_map or is_jupyter():
            _problem_name_to_problem_map[problem_name] = problem
        else:
            assert _problem_name_to_problem_map[problem_name] is problem


def get_problem_from_problem_name(problem_name):
    assert problem_name in _problem_name_to_problem_map
    return _problem_name_to_problem_map[problem_name]


_problem_name_to_problem_map = Cache()
