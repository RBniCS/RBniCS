# Copyright (C) 2015-2016 by the RBniCS authors
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

from dolfin import Function
from RBniCS.backends.fenics.wrapping.function_copy import function_copy

def function_component(function, component, copy):
    if component is None:
        if copy is True:
            return function_copy(function)
        else:
            return function
    else:
        assert copy is True, "It is not possible to clear components without copying the vector"
        V = function.function_space()
        num_components = V.num_subspaces()
        assert (
            (num_components == 0 and component == None)
                or
            (num_components > 0 and (component == None or component < num_components))
        )
        function_component = Function(V) # zero by default
        assign(function_component.sub(c), function.sub(c))
        return function_component

