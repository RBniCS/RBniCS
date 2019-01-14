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

from rbnics.eim.utils.decorators.store_map_from_parametrized_expression_to_problem import add_to_map_from_parametrized_expression_to_problem, get_problem_from_parametrized_expression
from rbnics.utils.decorators import PreserveClassName

def StoreMapFromParametrizedOperatorsToProblem(ExactParametrizedFunctionsDecoratedProblem_DerivedClass):
            
    @PreserveClassName
    class StoreMapFromParametrizedOperatorsToProblem_Class(ExactParametrizedFunctionsDecoratedProblem_DerivedClass):
        
        def _init_operators(self):
            # Initialize operators as in Parent class
            ExactParametrizedFunctionsDecoratedProblem_DerivedClass._init_operators(self)
            
            # Populate map from parametrized operators to (this) problem
            for (term, operator) in self.operator.items():
                if operator is not None: # raised by assemble_operator if output computation is optional
                    for operator_q in operator:
                        add_to_map_from_parametrized_operator_to_problem(operator_q, self) # this will also add non-parametrized assembled operator to the storage
            
    # return value (a class) for the decorator
    return StoreMapFromParametrizedOperatorsToProblem_Class
    
def add_to_map_from_parametrized_operator_to_problem(operator, problem):
    add_to_map_from_parametrized_expression_to_problem(operator, problem)
    
def get_problem_from_parametrized_operator(operator):
    return get_problem_from_parametrized_expression(operator)
