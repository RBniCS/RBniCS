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

def StoreMapFromProblemToReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
            
    @PreserveClassName
    class StoreMapFromProblemToReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            
            # Populate problem to reduced problem map
            add_to_map_from_problem_to_reduced_problem(truth_problem, self)
            
    # return value (a class) for the decorator
    return StoreMapFromProblemToReducedProblem_Class
    
def add_to_map_from_problem_to_reduced_problem(problem, reduced_problem):
    if problem not in _problem_to_reduced_problem_map:
        if hasattr(type(problem), "__is_exact__"):
            problem = problem.__decorated_problem__
        _problem_to_reduced_problem_map[problem] = reduced_problem
    else:
        assert _problem_to_reduced_problem_map[problem] is reduced_problem
    
def get_reduced_problem_from_problem(problem):
    assert problem in _problem_to_reduced_problem_map
    return _problem_to_reduced_problem_map[problem]
    
_problem_to_reduced_problem_map = Cache()
