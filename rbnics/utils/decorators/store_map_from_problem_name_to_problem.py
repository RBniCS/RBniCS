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

def StoreMapFromProblemNameToProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
            
    @PreserveClassName
    class StoreMapFromProblemNameToProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        
        def __init__(self, V, **kwargs):
            # Call the parent initialization
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
            
            # Populate problem name to problem map
            add_to_map_from_problem_name_to_problem(self.name(), self)
            
    # return value (a class) for the decorator
    return StoreMapFromProblemNameToProblem_Class
    
def add_to_map_from_problem_name_to_problem(problem_name, problem):
    if hasattr(type(problem), "__is_exact__"):
        assert type(problem).__is_exact__ is True
        problem_name = problem.__decorated_problem__.name()
        assert problem_name in _problem_name_to_problem_map
    else:
        if problem_name not in _problem_name_to_problem_map:
            _problem_name_to_problem_map[problem_name] = problem
        else:
            assert _problem_name_to_problem_map[problem_name] is problem
    
def get_problem_from_problem_name(problem_name):
    assert problem_name in _problem_name_to_problem_map
    return _problem_name_to_problem_map[problem_name]
    
_problem_name_to_problem_map = Cache()
