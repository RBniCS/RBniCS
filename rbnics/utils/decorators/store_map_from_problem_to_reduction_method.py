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
from rbnics.utils.decorators.preserve_class_name import PreserveClassName

def StoreMapFromProblemToReductionMethod(DifferentialProblemReductionMethod_DerivedClass):
            
    @PreserveClassName
    class StoreMapFromProblemToReductionMethod_Class(DifferentialProblemReductionMethod_DerivedClass):
        
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)
            
            # Populate problem to reduced problem map
            add_to_map_from_problem_to_reduction_method(truth_problem, self)
            
    # return value (a class) for the decorator
    return StoreMapFromProblemToReductionMethod_Class
    
def add_to_map_from_problem_to_reduction_method(problem, reduction_method):
    if problem not in _problem_to_reduction_method_map:
        if hasattr(type(problem), "__is_exact__"):
            problem = problem.__decorated_problem__
        _problem_to_reduction_method_map[problem] = reduction_method
    else:
        assert _problem_to_reduction_method_map[problem] is reduction_method
    
def get_reduction_method_from_problem(problem):
    assert problem in _problem_to_reduction_method_map
    return _problem_to_reduction_method_map[problem]
    
_problem_to_reduction_method_map = Cache()
