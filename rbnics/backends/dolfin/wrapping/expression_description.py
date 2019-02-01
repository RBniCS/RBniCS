# Copyright (C) 2015-2019 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

from dolfin import Constant
from rbnics.utils.decorators import get_problem_from_solution, get_problem_from_solution_dot

def basic_expression_description(backend, wrapping):
    def _basic_expression_description(expression):
        visited = set()
        coefficients_repr = dict()
        for n in wrapping.expression_iterator(expression):
            if n in visited:
                continue
            if hasattr(n, "_cppcode"):
                coefficients_repr[n] = str(n._cppcode)
                visited.add(n)
            elif wrapping.is_problem_solution_type(n):
                if wrapping.is_problem_solution(n):
                    (preprocessed_n, component, truth_solution) = wrapping.solution_identify_component(n)
                    problem = get_problem_from_solution(truth_solution)
                    coefficients_repr[preprocessed_n] = "solution of " + str(problem.name())
                elif wrapping.is_problem_solution_dot(n):
                    (preprocessed_n, component, truth_solution_dot) = wrapping.solution_dot_identify_component(n)
                    problem = get_problem_from_solution_dot(truth_solution_dot)
                    coefficients_repr[preprocessed_n] = "solution_dot of " + str(problem.name())
                else:
                    (preprocessed_n, component, problem) = wrapping.get_auxiliary_problem_for_non_parametrized_function(n)
                    coefficients_repr[preprocessed_n] = "non parametrized function associated to auxiliary problem " + str(problem.name())
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

from rbnics.backends.dolfin.wrapping.expression_iterator import expression_iterator
from rbnics.backends.dolfin.wrapping.get_auxiliary_problem_for_non_parametrized_function import get_auxiliary_problem_for_non_parametrized_function
from rbnics.backends.dolfin.wrapping.is_problem_solution import is_problem_solution
from rbnics.backends.dolfin.wrapping.is_problem_solution_dot import is_problem_solution_dot
from rbnics.backends.dolfin.wrapping.is_problem_solution_type import is_problem_solution_type
from rbnics.backends.dolfin.wrapping.solution_dot_identify_component import solution_dot_identify_component
from rbnics.backends.dolfin.wrapping.solution_identify_component import solution_identify_component
from rbnics.backends.dolfin.wrapping.solution_iterator import solution_iterator
from rbnics.utils.decorators import ModuleWrapper
backend = ModuleWrapper()
wrapping = ModuleWrapper(expression_iterator, is_problem_solution, is_problem_solution_dot, is_problem_solution_type, solution_dot_identify_component, solution_identify_component, solution_iterator, get_auxiliary_problem_for_non_parametrized_function=get_auxiliary_problem_for_non_parametrized_function)
expression_description = basic_expression_description(backend, wrapping)
