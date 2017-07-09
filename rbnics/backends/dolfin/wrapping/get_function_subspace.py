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

from dolfin import Function, FunctionSpace

def get_function_subspace(function_space__or__function, component):
    if isinstance(function_space__or__function, Function):
        function = function_space__or__function
        return get_function_subspace(function.function_space(), component)
    else:
        assert isinstance(function_space__or__function, FunctionSpace)
        function_space = function_space__or__function
        assert isinstance(component, (int, str, tuple))
        assert not isinstance(component, list), "dolfin does not handle yet the case of a list of components"
        if isinstance(component, tuple):
            return function_space.extract_sub_space(component).collapse()
        else:
            return function_space.sub(component).collapse()

