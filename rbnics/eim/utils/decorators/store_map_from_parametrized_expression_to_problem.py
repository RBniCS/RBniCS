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

from rbnics.utils.cache import Cache
from rbnics.utils.decorators import PreserveClassName

def StoreMapFromParametrizedExpressionToProblem(EIMApproximation_DerivedClass):
            
    @PreserveClassName
    class StoreMapFromParametrizedExpressionToProblem_Class(EIMApproximation_DerivedClass):
        
        def __init__(self, truth_problem, parametrized_expression, folder_prefix, basis_generation):
            # Call the parent initialization
            EIMApproximation_DerivedClass.__init__(self, truth_problem, parametrized_expression, folder_prefix, basis_generation)
            
            # Populate problem name to problem map
            add_to_map_from_parametrized_expression_to_problem(parametrized_expression, truth_problem)
            
    # return value (a class) for the decorator
    return StoreMapFromParametrizedExpressionToProblem_Class
    
def add_to_map_from_parametrized_expression_to_problem(parametrized_expression, problem):
    if hasattr(type(problem), "__is_exact__"):
        problem = problem.__decorated_problem__
    if parametrized_expression not in _parametrized_expression_to_problem_map:
        _parametrized_expression_to_problem_map[parametrized_expression] = problem
    else:
        assert _parametrized_expression_to_problem_map[parametrized_expression] is problem
    
def get_problem_from_parametrized_expression(parametrized_expression):
    assert parametrized_expression in _parametrized_expression_to_problem_map
    return _parametrized_expression_to_problem_map[parametrized_expression]
    
_parametrized_expression_to_problem_map = Cache()
