# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import hashlib
from ufl.core.multiindex import Index as MuteIndex, MultiIndex
from ufl.corealg.traversal import traverse_unique_terminals
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


def basic_expression_name(backend, wrapping):

    def _basic_expression_name(expression):
        str_repr = ""
        coefficients_replacement = dict()
        # Preprocess indices first, as their numeric value might change from run to run, but they
        # are always sorted the same way
        indices = set()
        min_index = None
        for t in traverse_unique_terminals(expression):
            if isinstance(t, MultiIndex):
                for i in t.indices():
                    if isinstance(i, MuteIndex):
                        if min_index is None or i.count() < min_index:
                            min_index = i.count()
                        indices.add(i)
        for i in indices:
            coefficients_replacement[repr(i)] = "MuteIndexRBniCS(" + str(i.count() - min_index) + ")"
        # Process the expression
        visited = set()
        for n in wrapping.expression_iterator(expression):
            if n in visited:
                continue
            if isinstance(n, BaseExpression):
                assert isinstance(n, (CompiledExpression, Expression)), "Other expression types are not handled yet"
                if isinstance(n, Expression):
                    coefficients_replacement[repr(n)] = str(n._cppcode)
                    str_repr += repr(n._cppcode)
                elif isinstance(n, CompiledExpression):
                    assert hasattr(n, "f_no_upcast"), "Only the case of pulled back expressions is currently handled"
                    assert hasattr(n, "shape_parametrization_expression_on_subdomain_no_upcast"), (
                        "Only the case of pulled back expressions is currently handled")
                    coefficients_replacement[repr(n)] = (
                        "PullBackExpression("
                        + str(n.shape_parametrization_expression_on_subdomain_no_upcast._cppcode)
                        + ", " + str(n.f_no_upcast._cppcode) + ")")
                    str_repr += (
                        "PullBackExpression("
                        + repr(n.shape_parametrization_expression_on_subdomain_no_upcast._cppcode)
                        + ", " + repr(n.f_no_upcast._cppcode) + ")")
                visited.add(n)
            elif wrapping.is_problem_solution_type(n):
                if wrapping.is_problem_solution(n):
                    (preprocessed_n, component, truth_solution) = wrapping.solution_identify_component(n)
                    problem = get_problem_from_solution(truth_solution)
                    coefficients_replacement[repr(preprocessed_n)] = (
                        "solution of " + str(problem.name())
                        + " (exact problem decorator: " + str(hasattr(problem, "__is_exact__"))
                        + ", component: " + str(component) + ")")
                elif wrapping.is_problem_solution_dot(n):
                    (preprocessed_n, component, truth_solution_dot) = wrapping.solution_dot_identify_component(n)
                    problem = get_problem_from_solution_dot(truth_solution_dot)
                    coefficients_replacement[repr(preprocessed_n)] = (
                        "solution_dot of " + str(problem.name())
                        + " (exact problem decorator: " + str(hasattr(problem, "__is_exact__"))
                        + ", component: " + str(component) + ")")
                else:
                    (preprocessed_n, component,
                     problem) = wrapping.get_auxiliary_problem_for_non_parametrized_function(n)
                    coefficients_replacement[repr(preprocessed_n)] = (
                        "non parametrized function associated to auxiliary problem " + str(problem.name()))
                if len(component) == 1 and component[0] is not None:
                    coefficients_replacement[repr(preprocessed_n)] += ", component " + str(component[0])
                elif len(component) > 1:
                    coefficients_replacement[repr(preprocessed_n)] += ", component " + str(component)
                str_repr += coefficients_replacement[repr(preprocessed_n)]
                # Make sure to skip any parent solution related to this one
                visited.add(n)
                visited.add(preprocessed_n)
                for parent_n in wrapping.solution_iterator(preprocessed_n):
                    visited.add(parent_n)
            elif isinstance(n, Constant):
                vals = n.values()
                coefficients_replacement[repr(n)] = str(vals)
                str_repr += repr(str(vals))
                visited.add(n)
            else:
                str_repr += repr(n)
                visited.add(n)
        for key, value in coefficients_replacement.items():
            str_repr = str_repr.replace(key, value)
        hash_code = hashlib.sha1(str_repr.encode("utf-8")).hexdigest()
        return hash_code

    return _basic_expression_name


backend = ModuleWrapper()
wrapping = ModuleWrapper(
    expression_iterator, is_problem_solution, is_problem_solution_dot, is_problem_solution_type,
    solution_dot_identify_component, solution_identify_component, solution_iterator,
    get_auxiliary_problem_for_non_parametrized_function=get_auxiliary_problem_for_non_parametrized_function)
expression_name = basic_expression_name(backend, wrapping)
