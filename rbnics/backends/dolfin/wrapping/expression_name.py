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
from dolfin import __version__ as dolfin_version, Constant, Function
from ufl.core.multiindex import FixedIndex, Index, MultiIndex
from ufl.indexed import Indexed
import hashlib
import rbnics.backends.dolfin
from rbnics.utils.decorators import get_problem_from_solution

def expression_name(expression, backend=None):
    if backend is None:
        backend = rbnics.backends.dolfin
    
    str_repr = ""
    visited = set()
    coefficients_replacement = dict()
    for n in backend.wrapping.expression_iterator(expression):
        if n in visited:
            continue
        n = _preprocess_indexed(n, coefficients_replacement, str_repr)
        if hasattr(n, "cppcode"):
            coefficients_replacement[repr(n)] = str(n.cppcode)
            str_repr += repr(n.cppcode)
            visited.add(n)
        elif backend.wrapping.is_problem_solution_or_problem_solution_component_type(n):
            if backend.wrapping.is_problem_solution_or_problem_solution_component(n):
                (preprocessed_n, component, truth_solution) = backend.wrapping.solution_identify_component(n)
                problem = get_problem_from_solution(truth_solution)
            else:
                (problem, component) = backend.wrapping.get_auxiliary_problem_for_non_parametrized_function(n)
                preprocessed_n = n
            coefficients_replacement[repr(preprocessed_n)] = str(type(problem).__name__) + str(component)
            str_repr += repr(type(problem).__name__) + str(component)
            # Make sure to skip any parent solution related to this one
            visited.add(n)
            visited.add(preprocessed_n)
            for parent_n in backend.wrapping.solution_iterator(preprocessed_n):
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
    for key, value in coefficients_replacement.iteritems():
        str_repr = str_repr.replace(key, value)
    hash_code = hashlib.sha1(
                    (str_repr + dolfin_version).encode("utf-8")
                ).hexdigest() # similar to dolfin/compilemodules/compilemodule.py
    return hash_code
    
def _preprocess_indexed(n, coefficients_replacement, str_repr):
    if isinstance(n, Indexed):
        assert len(n.ufl_operands) == 2
        assert isinstance(n.ufl_operands[1], MultiIndex)
        index_id = 0
        for index in n.ufl_operands[1].indices():
            assert isinstance(index, (FixedIndex, Index))
            if isinstance(index, FixedIndex):
                str_repr += repr(str(index))
            elif isinstance(index, Index):
                if repr(index) not in coefficients_replacement:
                    coefficients_replacement[repr(index)] = "i_" + str(index_id)
                    index_id += 1
                str_repr += coefficients_replacement[repr(index)]
        return n.ufl_operands[0]
    else:
        return n
        
