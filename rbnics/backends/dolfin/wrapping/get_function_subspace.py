# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

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
