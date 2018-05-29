# Copyright (C) 2015-2018 by the RBniCS authors
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

import hashlib
from ufl.core.multiindex import MultiIndex
from ufl.indexed import Indexed
from ufl.tensors import ListTensor
from dolfin import Function
from rbnics.backends.dolfin.wrapping.is_problem_solution_or_problem_solution_component import _split_function
from rbnics.backends.dolfin.wrapping.solution_identify_component import _remove_mute_indices
from rbnics.utils.cache import cache

def basic_get_auxiliary_problem_for_non_parametrized_function(backend, wrapping):
    @cache
    def _basic_get_auxiliary_problem_for_non_parametrized_function(function):
        assert isinstance(function, (Function, Indexed, ListTensor))
        converted_function = _remove_mute_indices(function)
        assert isinstance(converted_function, (Function, Indexed, ListTensor))
        if isinstance(converted_function, Function):
            component = (None, )
        elif isinstance(converted_function, Indexed):
            assert len(converted_function.ufl_operands) == 2
            assert isinstance(converted_function.ufl_operands[0], Function)
            assert isinstance(converted_function.ufl_operands[1], MultiIndex)
            function_split_to_component = dict()
            function_split_to_function = dict()
            _split_function(converted_function.ufl_operands[0], function_split_to_component, function_split_to_function)
            assert converted_function in function_split_to_component
            assert converted_function in function_split_to_function
            component = function_split_to_component[converted_function]
            converted_function = function_split_to_function[converted_function]
        elif isinstance(converted_function, ListTensor):
            assert all(isinstance(component, Indexed) for component in converted_function.ufl_operands)
            assert all(len(component.ufl_operands) == 2 for component in converted_function.ufl_operands)
            assert all(isinstance(component.ufl_operands[0], Function) for component in converted_function.ufl_operands)
            assert all(isinstance(component.ufl_operands[1], MultiIndex) for component in converted_function.ufl_operands)
            assert all(component.ufl_operands[0] == converted_function.ufl_operands[-1].ufl_operands[0] for component in converted_function.ufl_operands)
            function_split_to_component = dict()
            function_split_to_function = dict()
            _split_function(converted_function.ufl_operands[-1].ufl_operands[0], function_split_to_component, function_split_to_function)
            assert converted_function in function_split_to_component
            assert converted_function in function_split_to_function
            component = function_split_to_component[converted_function]
            converted_function = function_split_to_function[converted_function]
        else:
            raise ValueError("Invalid function provided to get_auxiliary_problem_for_non_parametrized_function")
            
        # Only a V attribute and a name method are required
        class AuxiliaryProblemForNonParametrizedFunction(object):
            def __init__(self, converted_function):
                self.V = wrapping.get_function_space(converted_function)
                
            def name(self):
                return type(self).__name__
                
        # Change the name of the (local) class to (almost) uniquely identify the function.
        # Since the unique dolfin identifier f_** may change between runs, we use as identifiers
        # a combination of the norms, truncated to the first five significant figures.
        norm_1 = round_to_significant_figures(wrapping.get_function_norm(converted_function, "l1"), 5)
        norm_2 = round_to_significant_figures(wrapping.get_function_norm(converted_function, "l2"), 5)
        norm_inf = round_to_significant_figures(wrapping.get_function_norm(converted_function, "linf"), 5)
        AuxiliaryProblemForNonParametrizedFunction.__name__ = (
            "Function_" + hashlib.sha1(
                (norm_1 + norm_2 + norm_inf).encode("utf-8")
            ).hexdigest()
        )
        return (AuxiliaryProblemForNonParametrizedFunction(converted_function), component)
        
    def round_to_significant_figures(x, n):
        return "%.*e" % (n-1, x)
        
    return _basic_get_auxiliary_problem_for_non_parametrized_function
    
from rbnics.backends.dolfin.wrapping.get_function_norm import get_function_norm
from rbnics.backends.dolfin.wrapping.get_function_space import get_function_space
from rbnics.utils.decorators import ModuleWrapper
backend = ModuleWrapper()
wrapping = ModuleWrapper(get_function_norm, get_function_space)
get_auxiliary_problem_for_non_parametrized_function = basic_get_auxiliary_problem_for_non_parametrized_function(backend, wrapping)
