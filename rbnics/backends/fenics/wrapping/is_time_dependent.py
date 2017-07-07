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

from dolfin import Expression, Function
import rbnics.backends.fenics
from rbnics.utils.decorators import get_problem_from_solution

def is_time_dependent(expression_or_form, iterator, backend=None):
    if backend is None:
        backend = rbnics.backends.fenics
        
    visited = set()
    for node in iterator(expression_or_form):
        # ... parametrized expressions
        if isinstance(node, Expression) and "t" in node.user_parameters:
            return True
        # ... problem solutions related to nonlinear terms
        elif backend.wrapping.is_problem_solution_or_problem_solution_component_type(node):
            if backend.wrapping.is_problem_solution_or_problem_solution_component(node):
                (preprocessed_node, component, truth_solution) = backend.wrapping.solution_identify_component(node)
                truth_problem = get_problem_from_solution(truth_solution)
                if hasattr(truth_problem, "set_time"):
                    return True
            else:
                preprocessed_node = node
            # Make sure to skip any parent solution related to this one
            visited.add(node)
            visited.add(preprocessed_node)
            for parent_node in backend.wrapping.solution_iterator(preprocessed_node):
                visited.add(parent_node)
    return False
    
