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

from rbnics.backends.dolfin.wrapping.is_problem_solution import _remove_all_indices, _split_function
from rbnics.utils.cache import Cache
from rbnics.utils.decorators.store_map_from_solution_dot_to_problem import _solution_dot_to_problem_map

def is_problem_solution_dot(node):
    _prepare_solution_dot_split_storage()
    node = _remove_all_indices(node)
    return node in _solution_dot_split_to_component
    
def _prepare_solution_dot_split_storage():
    for solution_dot in _solution_dot_to_problem_map:
        if solution_dot not in _solution_dot_split_to_component:
            assert solution_dot not in _solution_dot_split_to_solution_dot
            _split_function(solution_dot, _solution_dot_split_to_component, _solution_dot_split_to_solution_dot)
    
_solution_dot_split_to_component = Cache()
_solution_dot_split_to_solution_dot = Cache()
