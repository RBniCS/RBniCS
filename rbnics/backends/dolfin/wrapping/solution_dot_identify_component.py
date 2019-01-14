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

from rbnics.backends.dolfin.wrapping.is_problem_solution_dot import _solution_dot_split_to_component, _solution_dot_split_to_solution_dot
from rbnics.backends.dolfin.wrapping.solution_identify_component import _remove_mute_indices

def solution_dot_identify_component(node):
    node = _remove_mute_indices(node)
    return _solution_dot_identify_component(node)
    
def _solution_dot_identify_component(node):
    assert node in _solution_dot_split_to_component
    assert node in _solution_dot_split_to_solution_dot
    return (node, _solution_dot_split_to_component[node], _solution_dot_split_to_solution_dot[node])
