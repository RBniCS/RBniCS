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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from numpy import zeros
from dolfin import Constant, Function
from ufl.corealg.traversal import traverse_unique_terminals
from RBniCS.utils.decorators import get_problem_from_solution, is_problem_solution

def get_expression_description(expression):
    coefficients_repr = {}
    for n in traverse_unique_terminals(expression):
        if hasattr(n, "cppcode"):
            coefficients_repr[n] = str(n.cppcode)
        elif isinstance(n, Function) and is_problem_solution(n):
            problem = get_problem_from_solution(n)
            coefficients_repr[n] = "solution of " + str(type(problem).__name__)
        elif isinstance(n, Constant):
            x = zeros(1)
            vals = zeros(n.value_size())
            n.eval(vals, x)
            if len(vals) == 1:
                coefficients_repr[n] = str(vals[0])
            else:
                coefficients_repr[n] = str(vals.reshape(n.ufl_shape))
        else:
            assert not str(n).startswith("f_")
    return coefficients_repr
