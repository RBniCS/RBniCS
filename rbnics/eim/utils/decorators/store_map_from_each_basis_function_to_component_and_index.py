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

from rbnics.backends.abstract import BasisFunctionsMatrix as AbstractBasisFunctionsMatrix
from rbnics.utils.cache import Cache
from rbnics.utils.decorators import dict_of, overload, PreserveClassName
from rbnics.utils.test import PatchInstanceMethod

def StoreMapFromEachBasisFunctionToComponentAndIndex(ExactParametrizedFunctionsDecoratedReducedProblem_DerivedClass):
            
    @PreserveClassName
    class StoreMapFromEachBasisFunctionToComponentAndIndex_Class(ExactParametrizedFunctionsDecoratedReducedProblem_DerivedClass):
        
        def _init_basis_functions(self, current_stage="online"):
            # Initialize basis functions as in Parent class
            ExactParametrizedFunctionsDecoratedReducedProblem_DerivedClass._init_basis_functions(self, current_stage)
            
            # Patch BasisFunctionsMatrix._update_component_name_to_basis_component_length so that it also updates the map
            # from each basis function to component and index after BasisFunctionsMatrix.enrich() has been called.
            if not hasattr(self.basis_functions, "_update_component_name_to_basis_component_length_patched"):
                original_update_component_name_to_basis_component_length = self.basis_functions._update_component_name_to_basis_component_length
                
                @overload(AbstractBasisFunctionsMatrix, None)
                def patched_update_component_name_to_basis_component_length(self_, component):
                    assert len(self_._components) == 1
                    assert len(self_._components_name) == 1
                    component_0 = self_._components_name[0]
                    _add_new_basis_functions_to_map_from_basis_function_to_component_and_index(self_, component_0)
                    
                @overload(AbstractBasisFunctionsMatrix, str)
                def patched_update_component_name_to_basis_component_length(self_, component):
                    _add_new_basis_functions_to_map_from_basis_function_to_component_and_index(self_, component)
                    
                @overload(AbstractBasisFunctionsMatrix, dict_of(str, str))
                def patched_update_component_name_to_basis_component_length(self_, component):
                    assert len(component) == 1
                    for (_, component_to) in component.items():
                        break
                    assert component_to in self_._components
                    _add_new_basis_functions_to_map_from_basis_function_to_component_and_index(self_, component_to)
                    
                def _add_new_basis_functions_to_map_from_basis_function_to_component_and_index(self_, component):
                    old_component_length = self_._component_name_to_basis_component_length[component]
                    original_update_component_name_to_basis_component_length(component)
                    new_component_length = self_._component_name_to_basis_component_length[component]
                    for index in range(old_component_length, new_component_length):
                        add_to_map_from_basis_function_to_component_and_index(self_._components[component][index], component, index)
                    
                # Apply patch
                PatchInstanceMethod(self.basis_functions, "_update_component_name_to_basis_component_length", patched_update_component_name_to_basis_component_length).patch()
                self.basis_functions._update_component_name_to_basis_component_length_patched = True
            
    # return value (a class) for the decorator
    return StoreMapFromEachBasisFunctionToComponentAndIndex_Class
    
def add_to_map_from_basis_function_to_component_and_index(basis_function, component, index):
    if basis_function not in _basis_function_to_component_and_index_map:
        _basis_function_to_component_and_index_map[basis_function] = (component, index)
    else:
        assert (component, index) == _basis_function_to_component_and_index_map[basis_function]
    
def get_component_and_index_from_basis_function(basis_function):
    assert basis_function in _basis_function_to_component_and_index_map
    return _basis_function_to_component_and_index_map[basis_function]
    
_basis_function_to_component_and_index_map = Cache()
