# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.eim.utils.decorators.store_map_from_parametrized_expression_to_problem import (
    add_to_map_from_parametrized_expression_to_problem, get_problem_from_parametrized_expression)
from rbnics.utils.decorators import PreserveClassName


def StoreMapFromParametrizedOperatorsToProblem(ExactParametrizedFunctionsDecoratedProblem_DerivedClass):

    @PreserveClassName
    class StoreMapFromParametrizedOperatorsToProblem_Class(ExactParametrizedFunctionsDecoratedProblem_DerivedClass):

        def _init_operators(self):
            # Initialize operators as in Parent class
            ExactParametrizedFunctionsDecoratedProblem_DerivedClass._init_operators(self)

            # Populate map from parametrized operators to (this) problem
            for (term, operator) in self.operator.items():
                if operator is not None:  # raised by assemble_operator if output computation is optional
                    for operator_q in operator:
                        add_to_map_from_parametrized_operator_to_problem(operator_q, self)
                        # this will also add non-parametrized assembled operator to the storage

    # return value (a class) for the decorator
    return StoreMapFromParametrizedOperatorsToProblem_Class


def add_to_map_from_parametrized_operator_to_problem(operator, problem):
    add_to_map_from_parametrized_expression_to_problem(operator, problem)


def get_problem_from_parametrized_operator(operator):
    return get_problem_from_parametrized_expression(operator)
