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

from rbnics.utils.cache import Cache
from rbnics.utils.decorators import PreserveClassName
from rbnics.utils.test import PatchInstanceMethod

def StoreMapFromBasisFunctionsToReducedProblem(ExactParametrizedFunctionsDecoratedReducedProblem_DerivedClass):
            
    @PreserveClassName
    class StoreMapFromBasisFunctionsToReducedProblem_Class(ExactParametrizedFunctionsDecoratedReducedProblem_DerivedClass):
    
        def _init_basis_functions(self, current_stage="online"):
            # Initialize basis functions as in Parent class
            ExactParametrizedFunctionsDecoratedReducedProblem_DerivedClass._init_basis_functions(self, current_stage)
            
            # Populate basis functions to reduced problem map
            add_to_map_from_basis_functions_to_reduced_problem(self.basis_functions, self)
            
            # Add basis functions matrix obtained through sub components to the map as well, by patching
            # BasisFunctionsMatrix._precompute_sub_components
            if not hasattr(self.basis_functions, "_precompute_sub_components_patched"):
                original_precompute_sub_components = self.basis_functions._precompute_sub_components
                def patched_precompute_sub_components(self_, sub_components):
                    output = original_precompute_sub_components(sub_components)
                    add_to_map_from_basis_functions_to_reduced_problem(output, self)
                    return output
                # Apply patch
                PatchInstanceMethod(self.basis_functions, "_precompute_sub_components", patched_precompute_sub_components).patch()
                self.basis_functions._precompute_sub_components_patched = True
            
    # return value (a class) for the decorator
    return StoreMapFromBasisFunctionsToReducedProblem_Class
    
def add_to_map_from_basis_functions_to_reduced_problem(basis_functions, reduced_problem):
    if basis_functions not in _basis_functions_to_reduced_problem_map:
        _basis_functions_to_reduced_problem_map[basis_functions] = reduced_problem
    else:
        assert _basis_functions_to_reduced_problem_map[basis_functions] is reduced_problem
    
def get_reduced_problem_from_basis_functions(basis_functions):
    assert basis_functions in _basis_functions_to_reduced_problem_map
    return _basis_functions_to_reduced_problem_map[basis_functions]
    
_basis_functions_to_reduced_problem_map = Cache()
