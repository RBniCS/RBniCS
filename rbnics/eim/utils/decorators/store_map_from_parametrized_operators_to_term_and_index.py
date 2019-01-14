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

def StoreMapFromParametrizedOperatorsToTermAndIndex(ExactParametrizedFunctionsDecoratedProblem_DerivedClass):
            
    @PreserveClassName
    class StoreMapFromParametrizedOperatorsToTermAndIndex_Class(ExactParametrizedFunctionsDecoratedProblem_DerivedClass):
        
        def _init_operators(self):
            # Initialize operators as in Parent class
            ExactParametrizedFunctionsDecoratedProblem_DerivedClass._init_operators(self)
            
            # Populate map from parametrized operators to (this) problem
            for (term, operator) in self.operator.items():
                if operator is not None: # raised by assemble_operator if output computation is optional
                    for (q, operator_q) in enumerate(operator):
                        add_to_map_from_parametrized_operator_to_term_and_index(operator_q, term, q)
            
    # return value (a class) for the decorator
    return StoreMapFromParametrizedOperatorsToTermAndIndex_Class
    
def add_to_map_from_parametrized_operator_to_term_and_index(operator, term, index):
    if operator not in _parametrized_operator_to_term_and_index_map:
        _parametrized_operator_to_term_and_index_map[operator] = (term, index)
    else:
        # for simple problems the same operator may correspond to more than one term, we only care about one
        # of them anyway since we are going to use this function to only export the term name
        pass
    
def get_term_and_index_from_parametrized_operator(operator):
    assert operator in _parametrized_operator_to_term_and_index_map
    return _parametrized_operator_to_term_and_index_map[operator]
    
_parametrized_operator_to_term_and_index_map = Cache()
