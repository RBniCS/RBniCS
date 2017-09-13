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

def basic_is_parametrized(backend, wrapping):
    def _basic_is_parametrized(expression_or_form, iterator):
        for node in iterator(expression_or_form):
            # ... parametrized expressions
            if isinstance(node, Expression) and "mu_0" in node.user_parameters:
                return True
            # ... problem solutions related to nonlinear terms
            elif wrapping.is_problem_solution_or_problem_solution_component_type(node):
                if wrapping.is_problem_solution_or_problem_solution_component(node):
                    return True
        return False
    return _basic_is_parametrized
    
from rbnics.backends.dolfin.wrapping.is_problem_solution_or_problem_solution_component import is_problem_solution_or_problem_solution_component
from rbnics.backends.dolfin.wrapping.is_problem_solution_or_problem_solution_component_type import is_problem_solution_or_problem_solution_component_type
from rbnics.utils.decorators import ModuleWrapper
backend = ModuleWrapper()
wrapping = ModuleWrapper(is_problem_solution_or_problem_solution_component, is_problem_solution_or_problem_solution_component_type)
is_parametrized = basic_is_parametrized(backend, wrapping)
