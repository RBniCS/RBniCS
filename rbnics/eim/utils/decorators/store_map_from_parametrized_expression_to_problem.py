# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numbers import Number
from rbnics.utils.decorators import PreserveClassName


def StoreMapFromParametrizedExpressionToProblem(EIMApproximation_DerivedClass):

    @PreserveClassName
    class StoreMapFromParametrizedExpressionToProblem_Class(EIMApproximation_DerivedClass):

        def __init__(self, truth_problem, parametrized_expression, folder_prefix, basis_generation):
            # Call the parent initialization
            EIMApproximation_DerivedClass.__init__(
                self, truth_problem, parametrized_expression, folder_prefix, basis_generation)

            # Populate problem name to problem map
            add_to_map_from_parametrized_expression_to_problem(parametrized_expression, truth_problem)

    # return value (a class) for the decorator
    return StoreMapFromParametrizedExpressionToProblem_Class


def add_to_map_from_parametrized_expression_to_problem(parametrized_expression, problem):
    if hasattr(type(problem), "__is_exact__"):
        problem = problem.__decorated_problem__
    if not isinstance(parametrized_expression, Number):
        if not hasattr(parametrized_expression, "_problem"):
            setattr(parametrized_expression, "_problem", problem)
        else:
            assert parametrized_expression._problem is problem


def get_problem_from_parametrized_expression(parametrized_expression):
    assert hasattr(parametrized_expression, "_problem")
    return parametrized_expression._problem
