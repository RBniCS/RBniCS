# Copyright (C) 2015-2017 by the RBniCS authors
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

from numpy import zeros
from dolfin import Constant, Function
import rbnics.backends.fenics
from rbnics.utils.decorators import get_problem_from_solution

def expression_description(expression, backend=None):
    if backend is None:
        backend = rbnics.backends.fenics
    
    visited = set()
    coefficients_repr = dict()
    for n in backend.wrapping.expression_iterator(expression):
        if n in visited:
            continue
        if hasattr(n, "cppcode"):
            coefficients_repr[n] = str(n.cppcode)
            visited.add(n)
        elif backend.wrapping.is_problem_solution_or_problem_solution_component_type(n):
            if backend.wrapping.is_problem_solution_or_problem_solution_component(n):
                (preprocessed_n, component, truth_solution) = backend.wrapping.solution_identify_component(n)
                problem = get_problem_from_solution(truth_solution)
            else:
                (problem, component) = backend.wrapping.get_auxiliary_problem_for_non_parametrized_function(n)
                preprocessed_n = n
            coefficients_repr[preprocessed_n] = "solution of " + str(type(problem).__name__)
            if len(component) is 1 and component[0] is not None:
                coefficients_repr[preprocessed_n] += ", component " + str(component[0])
            elif len(component) > 1:
                coefficients_repr[preprocessed_n] += ", component " + str(component)
            # Make sure to skip any parent solution related to this one
            visited.add(n)
            visited.add(preprocessed_n)
            for parent_n in backend.wrapping.solution_iterator(preprocessed_n):
                visited.add(parent_n)
        elif isinstance(n, Constant):
            x = zeros(1)
            vals = zeros(n.value_size())
            n.eval(vals, x)
            if len(vals) == 1:
                coefficients_repr[n] = str(vals[0])
            else:
                coefficients_repr[n] = str(vals.reshape(n.ufl_shape))
            visited.add(n)
    return coefficients_repr
