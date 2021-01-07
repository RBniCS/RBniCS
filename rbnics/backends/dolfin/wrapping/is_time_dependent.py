# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin.function.expression import BaseExpression
from rbnics.backends.dolfin.wrapping.is_problem_solution import is_problem_solution
from rbnics.backends.dolfin.wrapping.is_problem_solution_dot import is_problem_solution_dot
from rbnics.backends.dolfin.wrapping.is_problem_solution_type import is_problem_solution_type
from rbnics.backends.dolfin.wrapping.solution_identify_component import solution_identify_component
from rbnics.backends.dolfin.wrapping.solution_iterator import solution_iterator
from rbnics.backends.dolfin.wrapping.pull_back_to_reference_domain import (
    is_pull_back_expression, is_pull_back_expression_time_dependent)
from rbnics.utils.decorators import get_problem_from_solution, ModuleWrapper


def basic_is_time_dependent(backend, wrapping):

    def _basic_is_time_dependent(expression_or_form, iterator):
        for node in iterator(expression_or_form):
            # ... parametrized expressions
            if isinstance(node, BaseExpression):
                if is_pull_back_expression(node) and is_pull_back_expression_time_dependent(node):
                    return True
                else:
                    parameters = node._parameters
                    if "t" in parameters:
                        return True
            # ... problem solutions related to nonlinear terms
            elif wrapping.is_problem_solution_type(node):
                if wrapping.is_problem_solution(node):
                    (preprocessed_node, component, truth_solution) = wrapping.solution_identify_component(node)
                    truth_problem = get_problem_from_solution(truth_solution)
                    if hasattr(truth_problem, "set_time"):
                        return True
                elif wrapping.is_problem_solution_dot(node):
                    return True
        return False

    return _basic_is_time_dependent


backend = ModuleWrapper()
wrapping = ModuleWrapper(is_problem_solution, is_problem_solution_dot, is_problem_solution_type,
                         solution_identify_component, solution_iterator)
is_time_dependent = basic_is_time_dependent(backend, wrapping)
