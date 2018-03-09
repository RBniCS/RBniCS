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

from rbnics.utils.decorators import PreserveClassName

def StoreMapFromBasisFunctionsToReducedProblem(ExactParametrizedFunctionsDecoratedReducedProblem_DerivedClass):
            
    @PreserveClassName
    class StoreMapFromBasisFunctionsToReducedProblem_Class(ExactParametrizedFunctionsDecoratedReducedProblem_DerivedClass):
    
        def _init_basis_functions(self, current_stage="online"):
            # Initialize basis functions as in Parent class
            ExactParametrizedFunctionsDecoratedReducedProblem_DerivedClass._init_basis_functions(self, current_stage)
            
            # Populate basis functions to reduced problem map
            add_to_map_from_basis_functions_to_reduced_problem(self.basis_functions, self)
            
            # Patch BasisFunctionsMatrix's __getitem__ to store the sub components basis function before returning
            def patch_getitem(input_):
                Type = type(input_) # note that we need to patch the type (and not the instance) because __getitem__ is a magic method
                if not hasattr(Type, "getitem_patched_for_list_of_str"):
                    original_getitem = Type.__getitem__
                    def patched_getitem(self_, key):
                        output = original_getitem(self_, key)
                        if isinstance(key, list) and all(isinstance(k, str) for k in key):
                            add_to_map_from_basis_functions_to_reduced_problem(output, self)
                        return output
                    # Apply patch
                    Type.__getitem__ = patched_getitem
                    Type.getitem_patched_for_list_of_str = True
            patch_getitem(self.basis_functions)
            
    # return value (a class) for the decorator
    return StoreMapFromBasisFunctionsToReducedProblem_Class
    
def add_to_map_from_basis_functions_to_reduced_problem(basis_functions, reduced_problem):
    if basis_functions not in _basis_functions_to_reduced_problem_map:
        _basis_functions_to_reduced_problem_map[basis_functions] = reduced_problem
    else:
        assert reduced_problem is _basis_functions_to_reduced_problem_map[basis_functions]
    
def get_reduced_problem_from_basis_functions(basis_functions):
    assert basis_functions in _basis_functions_to_reduced_problem_map
    return _basis_functions_to_reduced_problem_map[basis_functions]
    
_basis_functions_to_reduced_problem_map = dict()
