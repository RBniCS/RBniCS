# Copyright (C) 2015-2017 by the RBniCS authors
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

from rbnics.utils.decorators import Extends

def StoreMapFromParametrizedExpressionToEIMApproximation(EIMApproximation_DerivedClass):
            
    @Extends(EIMApproximation_DerivedClass, preserve_class_name=True)
    class StoreMapFromParametrizedExpressionToEIMApproximation_Class(EIMApproximation_DerivedClass):
        
        def __init__(self, truth_problem, parametrized_expression, folder_prefix, basis_generation):
            # Call the parent initialization
            EIMApproximation_DerivedClass.__init__(self, truth_problem, parametrized_expression, folder_prefix, basis_generation)
            
            # Populate problem name to problem map
            add_to_map_from_parametrized_expression_to_EIM_approximation(parametrized_expression, self)
            
    # return value (a class) for the decorator
    return StoreMapFromParametrizedExpressionToEIMApproximation_Class
    
def add_to_map_from_parametrized_expression_to_EIM_approximation(parametrized_expression, EIM_approximation):
    assert parametrized_expression not in _parametrized_expression_to_EIM_approximation_map
    _parametrized_expression_to_EIM_approximation_map[parametrized_expression] = EIM_approximation
    
def get_EIM_approximation_from_parametrized_expression(parametrized_expression):
    assert parametrized_expression in _parametrized_expression_to_EIM_approximation_map
    return _parametrized_expression_to_EIM_approximation_map[parametrized_expression]
    
_parametrized_expression_to_EIM_approximation_map = dict()

