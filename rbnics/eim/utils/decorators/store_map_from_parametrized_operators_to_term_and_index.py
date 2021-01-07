# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.cache import Cache
from rbnics.utils.decorators import PreserveClassName


def StoreMapFromParametrizedOperatorsToTermAndIndex(ExactParametrizedFunctionsDecoratedProblem_DerivedClass):

    @PreserveClassName
    class StoreMapFromParametrizedOperatorsToTermAndIndex_Class(
            ExactParametrizedFunctionsDecoratedProblem_DerivedClass):

        def _init_operators(self):
            # Initialize operators as in Parent class
            ExactParametrizedFunctionsDecoratedProblem_DerivedClass._init_operators(self)

            # Populate map from parametrized operators to (this) problem
            for (term, operator) in self.operator.items():
                if operator is not None:  # raised by assemble_operator if output computation is optional
                    for (q, operator_q) in enumerate(operator):
                        add_to_map_from_parametrized_operator_to_term_and_index(operator_q, term, q)

    # return value (a class) for the decorator
    return StoreMapFromParametrizedOperatorsToTermAndIndex_Class


def add_to_map_from_parametrized_operator_to_term_and_index(operator, term, index):
    if operator not in _parametrized_operator_to_term_and_index_map:
        _parametrized_operator_to_term_and_index_map[operator] = (term, index)
    else:
        # for simple problems the same operator may correspond to more than one term, we only care about one
        # of them anyway since we are going to use this function to only export the term name
        pass


def get_term_and_index_from_parametrized_operator(operator):
    assert operator in _parametrized_operator_to_term_and_index_map
    return _parametrized_operator_to_term_and_index_map[operator]


_parametrized_operator_to_term_and_index_map = Cache()
