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
from rbnics.utils.cache import Cache, cache

def basic_get_auxiliary_problem_for_non_parametrized_function(backend, wrapping):
    @cache # to skip preprocessing if node is queried multiple time
    def _basic_get_auxiliary_problem_for_non_parametrized_function(node):
        assert isinstance(node, (Function, Indexed, ListTensor))
        preprocessed_node = _remove_mute_indices(node)
        assert isinstance(preprocessed_node, (Function, Indexed, ListTensor))
        if isinstance(preprocessed_node, Function):
            function = preprocessed_node
            component = (None, )
        elif isinstance(preprocessed_node, Indexed):
            assert len(preprocessed_node.ufl_operands) == 2
            assert isinstance(preprocessed_node.ufl_operands[0], Function)
            assert isinstance(preprocessed_node.ufl_operands[1], MultiIndex)
            function_split_to_component = dict()
            function_split_to_function = dict()
            _split_function(preprocessed_node.ufl_operands[0], function_split_to_component, function_split_to_function)
            assert preprocessed_node in function_split_to_component
            assert preprocessed_node in function_split_to_function
            component = function_split_to_component[preprocessed_node]
            function = function_split_to_function[preprocessed_node]
        elif isinstance(preprocessed_node, ListTensor):
            assert all(isinstance(component, Indexed) for component in preprocessed_node.ufl_operands)
            assert all(len(component.ufl_operands) == 2 for component in preprocessed_node.ufl_operands)
            assert all(isinstance(component.ufl_operands[0], Function) for component in preprocessed_node.ufl_operands)
            assert all(isinstance(component.ufl_operands[1], MultiIndex) for component in preprocessed_node.ufl_operands)
            assert all(component.ufl_operands[0] == preprocessed_node.ufl_operands[-1].ufl_operands[0] for component in preprocessed_node.ufl_operands)
            function_split_to_component = dict()
            function_split_to_function = dict()
            _split_function(preprocessed_node.ufl_operands[-1].ufl_operands[0], function_split_to_component, function_split_to_function)
            assert preprocessed_node in function_split_to_component
            assert preprocessed_node in function_split_to_function
            component = function_split_to_component[preprocessed_node]
            function = function_split_to_function[preprocessed_node]
        else:
            raise ValueError("Invalid function provided to get_auxiliary_problem_for_non_parametrized_function")
            
        try:
            auxiliary_problem_for_non_parametrized_function = _auxiliary_problem_for_non_parametrized_function_cache[function]
        except KeyError:
            # Only a V attribute and a name method are required
            class AuxiliaryProblemForNonParametrizedFunction(object):
                def __init__(self, function):
                    self.V = wrapping.get_function_space(function)
                    
                def name(self):
                    return type(self).__name__
                    
            # Change the name of the (local) class to uniquely identify the function.
            AuxiliaryProblemForNonParametrizedFunction.__name__ += "_" + str(len(_auxiliary_problem_for_non_parametrized_function_cache))
            auxiliary_problem_for_non_parametrized_function = AuxiliaryProblemForNonParametrizedFunction(function)
            _auxiliary_problem_for_non_parametrized_function_cache[function] = auxiliary_problem_for_non_parametrized_function
            
        return (preprocessed_node, component, auxiliary_problem_for_non_parametrized_function)
        
    _auxiliary_problem_for_non_parametrized_function_cache = Cache() # over functions rather than nodes, as after preprocessing two nodes can be related to different components of the same function
        
    return _basic_get_auxiliary_problem_for_non_parametrized_function
    
from rbnics.backends.dolfin.wrapping.get_function_norm import get_function_norm
from rbnics.backends.dolfin.wrapping.get_function_space import get_function_space
from rbnics.utils.decorators import ModuleWrapper
backend = ModuleWrapper()
wrapping = ModuleWrapper(get_function_norm, get_function_space)
get_auxiliary_problem_for_non_parametrized_function = basic_get_auxiliary_problem_for_non_parametrized_function(backend, wrapping)
