# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import Function, FunctionSpace
from rbnics.utils.decorators import backend_for
from rbnics.backends.dolfin.wrapping.function_space import _convert_component_to_int, _enable_string_components

_Function_Type = Function


@backend_for("dolfin", inputs=(FunctionSpace, (str, None)))
def Function(V, component=None):
    if component is None:
        return _Function_Type(V)
    else:
        V = V.sub(component).collapse()
        return _Function_Type(V)


# Attach a Type() function
def Type():
    return _Function_Type


Function.Type = Type


# Make sure that _Function_Type.function_space() preserves component to index map
original__init__ = _Function_Type.__init__


def custom__init__(self, *args, **kwargs):
    if isinstance(args[0], FunctionSpace) and hasattr(args[0], "_component_to_index"):
        self._component_to_index = args[0]._component_to_index
    original__init__(self, *args, **kwargs)


_Function_Type.__init__ = custom__init__


original_function_space = _Function_Type.function_space


def custom_function_space(self):
    output = original_function_space(self)
    if hasattr(self, "_component_to_index"):
        _enable_string_components(self._component_to_index, output)
    return output


_Function_Type.function_space = custom_function_space


# Also make _Function_Type.sub() aware of string components
original_sub = _Function_Type.sub


def custom_sub(self, i, deepcopy=False):
    if hasattr(self, "_component_to_index"):
        i_int = _convert_component_to_int(self, i)
        assert isinstance(i_int, (int, tuple))
        if isinstance(i_int, int):
            return original_sub(self, i_int, deepcopy)
        else:
            output = self
            for sub_i in i_int:
                output = output.sub(sub_i, deepcopy)
            return output
    else:
        return original_sub(self, i, deepcopy)


_Function_Type.sub = custom_sub
