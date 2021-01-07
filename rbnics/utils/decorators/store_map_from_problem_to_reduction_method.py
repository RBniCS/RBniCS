# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.cache import Cache
from rbnics.utils.decorators.preserve_class_name import PreserveClassName
from rbnics.utils.jupyter import is_jupyter


def StoreMapFromProblemToReductionMethod(DifferentialProblemReductionMethod_DerivedClass):

    @PreserveClassName
    class StoreMapFromProblemToReductionMethod_Class(DifferentialProblemReductionMethod_DerivedClass):

        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)

            # Populate problem to reduced problem map
            add_to_map_from_problem_to_reduction_method(truth_problem, self)

    # return value (a class) for the decorator
    return StoreMapFromProblemToReductionMethod_Class


def add_to_map_from_problem_to_reduction_method(problem, reduction_method):
    if problem not in _problem_to_reduction_method_map or is_jupyter():
        if hasattr(type(problem), "__is_exact__"):
            problem = problem.__decorated_problem__
        _problem_to_reduction_method_map[problem] = reduction_method
    else:
        assert _problem_to_reduction_method_map[problem] is reduction_method


def get_reduction_method_from_problem(problem):
    assert problem in _problem_to_reduction_method_map
    return _problem_to_reduction_method_map[problem]


_problem_to_reduction_method_map = Cache()
