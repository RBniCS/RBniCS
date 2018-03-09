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

from rbnics.backends.abstract import BasisFunctionsMatrix as AbstractBasisFunctionsMatrix, FunctionsList as AbstractFunctionsList
from rbnics.utils.decorators import PreserveClassName

def StoreMapFromEachBasisFunctionToComponentAndIndex(ExactParametrizedFunctionsDecoratedReducedProblem_DerivedClass):
            
    @PreserveClassName
    class StoreMapFromEachBasisFunctionToComponentAndIndex_Class(ExactParametrizedFunctionsDecoratedReducedProblem_DerivedClass):
        
        def _init_basis_functions(self, current_stage="online"):
            # Initialize basis functions as in Parent class
            ExactParametrizedFunctionsDecoratedReducedProblem_DerivedClass._init_basis_functions(self, current_stage)
            
            # Patch BasisFunctionsMatrix's __getitem__ to store component and index before returning
            def patch_getitem(input_):
                Type = type(input_) # note that we need to patch the type (and not the instance) because __getitem__ is a magic method
                if not hasattr(Type, "getitem_patched_for_int_and_str"):
                    original_getitem = Type.__getitem__
                    def patched_getitem(self_, key):
                        output = original_getitem(self_, key)
                        if isinstance(key, int):
                            add_to_map_from_basis_function_to_component_and_index(output, None, key)
                        elif isinstance(key, str):
                            assert isinstance(self_, AbstractBasisFunctionsMatrix)
                            assert isinstance(output, AbstractFunctionsList)
                            patch_getitem(output)
                        return output
                    # Apply patch
                    Type.__getitem__ = patched_getitem
                    Type.getitem_patched_for_int_and_str = True
            patch_getitem(self.basis_functions)
            
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
    
_basis_function_to_component_and_index_map = dict()
