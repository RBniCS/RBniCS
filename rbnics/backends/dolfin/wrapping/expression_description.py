# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import CompiledExpression, Constant, Expression
from dolfin.function.expression import BaseExpression
from rbnics.backends.dolfin.wrapping.expression_iterator import expression_iterator
from rbnics.backends.dolfin.wrapping.get_auxiliary_problem_for_non_parametrized_function import (
    get_auxiliary_problem_for_non_parametrized_function)
from rbnics.backends.dolfin.wrapping.is_problem_solution import is_problem_solution
from rbnics.backends.dolfin.wrapping.is_problem_solution_dot import is_problem_solution_dot
from rbnics.backends.dolfin.wrapping.is_problem_solution_type import is_problem_solution_type
from rbnics.backends.dolfin.wrapping.solution_dot_identify_component import solution_dot_identify_component
from rbnics.backends.dolfin.wrapping.solution_identify_component import solution_identify_component
from rbnics.backends.dolfin.wrapping.solution_iterator import solution_iterator
from rbnics.utils.decorators import get_problem_from_solution, get_problem_from_solution_dot, ModuleWrapper


def basic_expression_description(backend, wrapping):

    def _basic_expression_description(expression):
        visited = set()
        coefficients_repr = dict()
        for n in wrapping.expression_iterator(expression):
            if n in visited:
                continue
            if isinstance(n, BaseExpression):
                assert isinstance(n, (CompiledExpression, Expression)), "Other expression types are not handled yet"
                if isinstance(n, Expression):
                    coefficients_repr[n] = str(n._cppcode)
                elif isinstance(n, CompiledExpression):
                    assert hasattr(n, "f_no_upcast"), "Only the case of pulled back expressions is currently handled"
                    assert hasattr(n, "shape_parametrization_expression_on_subdomain_no_upcast"), (
                        "Only the case of pulled back expressions is currently handled")
                    coefficients_repr[n] = (
                        "PullBackExpression("
                        + str(n.shape_parametrization_expression_on_subdomain_no_upcast._cppcode)
                        + ", " + str(n.f_no_upcast._cppcode) + ")")
                visited.add(n)
            elif wrapping.is_problem_solution_type(n):
                if wrapping.is_problem_solution(n):
                    (preprocessed_n, component, truth_solution) = wrapping.solution_identify_component(n)
                    problem = get_problem_from_solution(truth_solution)
                    coefficients_repr[preprocessed_n] = (
                        "solution of " + str(problem.name())
                        + " (exact problem decorator: " + str(hasattr(problem, "__is_exact__"))
                        + ", component: " + str(component) + ")")
                elif wrapping.is_problem_solution_dot(n):
                    (preprocessed_n, component, truth_solution_dot) = wrapping.solution_dot_identify_component(n)
                    problem = get_problem_from_solution_dot(truth_solution_dot)
                    coefficients_repr[preprocessed_n] = (
                        "solution_dot of " + str(problem.name())
                        + " (exact problem decorator: " + str(hasattr(problem, "__is_exact__"))
                        + ", component: " + str(component) + ")")
                else:
                    (preprocessed_n, component,
                     problem) = wrapping.get_auxiliary_problem_for_non_parametrized_function(n)
                    coefficients_repr[preprocessed_n] = (
                        "non parametrized function associated to auxiliary problem "
                        + str(problem.name()))
                if len(component) == 1 and component[0] is not None:
                    coefficients_repr[preprocessed_n] += ", component " + str(component[0])
                elif len(component) > 1:
                    coefficients_repr[preprocessed_n] += ", component " + str(component)
                # Make sure to skip any parent solution related to this one
                visited.add(n)
                visited.add(preprocessed_n)
                for parent_n in wrapping.solution_iterator(preprocessed_n):
                    visited.add(parent_n)
            elif isinstance(n, Constant):
                vals = n.values()
                if len(vals) == 1:
                    coefficients_repr[n] = str(vals[0])
                else:
                    coefficients_repr[n] = str(vals.reshape(n.ufl_shape))
                visited.add(n)
            else:
                visited.add(n)
        return coefficients_repr

    return _basic_expression_description


backend = ModuleWrapper()
wrapping = ModuleWrapper(
    expression_iterator, is_problem_solution, is_problem_solution_dot, is_problem_solution_type,
    solution_dot_identify_component, solution_identify_component, solution_iterator,
    get_auxiliary_problem_for_non_parametrized_function=get_auxiliary_problem_for_non_parametrized_function)
expression_description = basic_expression_description(backend, wrapping)
