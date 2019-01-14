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

from dolfin.function.expression import BaseExpression
from rbnics.backends.dolfin.wrapping.pull_back_to_reference_domain import is_pull_back_expression, is_pull_back_expression_parametrized

def basic_is_parametrized(backend, wrapping):
    def _basic_is_parametrized(expression_or_form, iterator):
        for node in iterator(expression_or_form):
            # ... parametrized expressions
            if isinstance(node, BaseExpression):
                if is_pull_back_expression(node) and is_pull_back_expression_parametrized(node):
                    return True
                else:
                    parameters = node._parameters
                    if "mu_0" in parameters:
                        return True
            # ... problem solutions related to nonlinear terms
            elif wrapping.is_problem_solution_type(node):
                if wrapping.is_problem_solution(node) or wrapping.is_problem_solution_dot(node):
                    return True
        return False
    return _basic_is_parametrized
    
from rbnics.backends.dolfin.wrapping.is_problem_solution import is_problem_solution
from rbnics.backends.dolfin.wrapping.is_problem_solution_dot import is_problem_solution_dot
from rbnics.backends.dolfin.wrapping.is_problem_solution_type import is_problem_solution_type
from rbnics.utils.decorators import ModuleWrapper
backend = ModuleWrapper()
wrapping = ModuleWrapper(is_problem_solution, is_problem_solution_dot, is_problem_solution_type)
is_parametrized = basic_is_parametrized(backend, wrapping)
