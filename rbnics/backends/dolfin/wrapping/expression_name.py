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
from dolfin import __version__ as dolfin_version, Constant
from ufl.core.multiindex import FixedIndex, Index, MultiIndex
from ufl.indexed import Indexed
import hashlib
from rbnics.utils.decorators import get_problem_from_solution

def basic_expression_name(backend, wrapping):
    def _basic_expression_name(expression):
        str_repr = ""
        visited = set()
        coefficients_replacement = dict()
        for n in wrapping.expression_iterator(expression):
            if n in visited:
                continue
            if hasattr(n, "cppcode"):
                coefficients_replacement[repr(n)] = str(n.cppcode)
                str_repr += repr(n.cppcode)
                visited.add(n)
            elif wrapping.is_problem_solution_or_problem_solution_component_type(n):
                if wrapping.is_problem_solution_or_problem_solution_component(n):
                    (preprocessed_n, component, truth_solution) = wrapping.solution_identify_component(n)
                    problem = get_problem_from_solution(truth_solution)
                else:
                    (problem, component) = wrapping.get_auxiliary_problem_for_non_parametrized_function(n)
                    preprocessed_n = n
                coefficients_replacement[repr(preprocessed_n)] = str(problem.name()) + str(component)
                str_repr += repr(problem.name()) + str(component)
                # Make sure to skip any parent solution related to this one
                visited.add(n)
                visited.add(preprocessed_n)
                for parent_n in wrapping.solution_iterator(preprocessed_n):
                    visited.add(parent_n)
            elif isinstance(n, Constant):
                x = zeros(1)
                vals = zeros(n.value_size())
                n.eval(vals, x)
                coefficients_replacement[repr(n)] = str(vals)
                str_repr += repr(str(vals))
                visited.add(n)
            else:
                str_repr += repr(n)
                visited.add(n)
        for key, value in coefficients_replacement.items():
            str_repr = str_repr.replace(key, value)
        hash_code = hashlib.sha1(
                        (str_repr + dolfin_version).encode("utf-8")
                    ).hexdigest() # similar to dolfin/compilemodules/compilemodule.py
        return hash_code
    
    return _basic_expression_name

from rbnics.backends.dolfin.wrapping.expression_iterator import expression_iterator
from rbnics.backends.dolfin.wrapping.get_auxiliary_problem_for_non_parametrized_function import get_auxiliary_problem_for_non_parametrized_function
from rbnics.backends.dolfin.wrapping.is_problem_solution_or_problem_solution_component import is_problem_solution_or_problem_solution_component
from rbnics.backends.dolfin.wrapping.is_problem_solution_or_problem_solution_component_type import is_problem_solution_or_problem_solution_component_type
from rbnics.backends.dolfin.wrapping.solution_identify_component import solution_identify_component
from rbnics.backends.dolfin.wrapping.solution_iterator import solution_iterator
from rbnics.utils.decorators import ModuleWrapper
backend = ModuleWrapper()
wrapping = ModuleWrapper(expression_iterator, is_problem_solution_or_problem_solution_component, is_problem_solution_or_problem_solution_component_type, solution_identify_component, solution_iterator, get_auxiliary_problem_for_non_parametrized_function=get_auxiliary_problem_for_non_parametrized_function)
expression_name = basic_expression_name(backend, wrapping)
