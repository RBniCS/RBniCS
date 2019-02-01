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

from dolfin import Function, FunctionSpace
from rbnics.utils.decorators import list_of, overload, tuple_of

@overload
def get_function_subspace(function: Function, component: (int, list_of(str), str, tuple_of(int))):
    return get_function_subspace(function.function_space(), component)

@overload
def get_function_subspace(function_space: FunctionSpace, component: (int, str)):
    return function_space.sub(component).collapse()
    
@overload
def get_function_subspace(function_space: FunctionSpace, component: list_of(str)):
    assert len(set([function_space.component_to_index(c) for c in component])) == 1
    output = function_space.sub(component[0]).collapse()
    output._component_to_index.clear()
    for c in component:
        output._component_to_index[c] = None
    output._index_to_components.clear()
    output._index_to_components[None] = component
    return output
    
@overload
def get_function_subspace(function_space: FunctionSpace, component: tuple_of(int)):
    return function_space.extract_sub_space(component).collapse()
