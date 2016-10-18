# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file 
#  @brief 
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.utils.decorators.extends import Extends
from RBniCS.utils.decorators.override import override

def StoreMapFromBasisFunctionsMatrixToReducedProblem(ReducedParametrizedProblem_DerivedClass):
            
    @Extends(ReducedParametrizedProblem_DerivedClass, preserve_class_name=True)
    class StoreMapFromBasisFunctionsMatrixToReducedProblem_Class(ReducedParametrizedProblem_DerivedClass):
        
        @override
        def __init__(self, truth_problem):
            # Call the parent initialization
            ReducedParametrizedProblem_DerivedClass.__init__(self, truth_problem)
            
            # Populate basis functions matrix to reduced problem map
            add_to_map_from_basis_functions_matrix_to_reduced_problem(self.Z, self)
            
    # return value (a class) for the decorator
    return StoreMapFromBasisFunctionsMatrixToReducedProblem_Class
    
def add_to_map_from_basis_functions_matrix_to_reduced_problem(Z, problem):
    assert Z not in _basis_functions_matrix_to_reduced_problem_map
    _basis_functions_matrix_to_reduced_problem_map[Z] = problem
    
def get_reduced_problem_from_basis_functions_matrix(Z):
    assert Z in _basis_functions_matrix_to_reduced_problem_map
    return _basis_functions_matrix_to_reduced_problem_map[Z]
    
_basis_functions_matrix_to_reduced_problem_map = dict()

