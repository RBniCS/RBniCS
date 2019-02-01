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

from collections import OrderedDict
import ufl
from dolfin import FunctionSpace
from rbnics.utils.test import AttachInstanceMethod, PatchInstanceMethod

original_FunctionSpace__init__ = FunctionSpace.__init__
def custom_FunctionSpace__init__(self, *args, **kwargs):
    if "components" in kwargs:
        components = kwargs["components"]
        del kwargs["components"]
    else:
        components = None
    original_FunctionSpace__init__(self, *args, **kwargs)
    if components is not None:
        _enable_string_components(components, self)
FunctionSpace.__init__ = custom_FunctionSpace__init__

def custom_FunctionSpace__hash__(self):
    return ufl.FunctionSpace.__hash__(self)
FunctionSpace.__hash__ = custom_FunctionSpace__hash__
    
def _enable_string_components(components, function_space):
    _init_component_to_index(components, function_space)
    
    original_sub = function_space.sub
    def custom_sub(self_, i):
        assert i is not None
        i_int = _convert_component_to_int(self_, i)
        if i_int is None:
            def custom_collapse(self_, collapsed_dofs=False):
                assert not collapsed_dofs
                return self_
            PatchInstanceMethod(self_, "collapse", custom_collapse).patch()
            return self_
        assert isinstance(i_int, (int, tuple))
        if isinstance(i_int, int):
            output = original_sub(i_int)
        else:
            output = self_.extract_sub_space(i_int)
        if isinstance(i, str):
            components = OrderedDict()
            components[i] = None
        else:
            components = OrderedDict()
            if (
                len(self_._index_to_components) == 1
                    and
                None in self_._index_to_components
            ):
                for c in self_._index_to_components[None]:
                    components[c] = None
            else:
                for c in self_.index_to_components(i):
                    components[c] = None
        _enable_string_components(components, output)
        return output
    PatchInstanceMethod(function_space, "sub", custom_sub).patch()
    
    _preserve_root_space_after_sub(function_space, None)
    
    original_extract_sub_space = function_space.extract_sub_space
    def custom_extract_sub_space(self_, i):
        i_int = _convert_component_to_int(self_, i)
        output = original_extract_sub_space(i_int)
        if isinstance(i, str):
            components = OrderedDict()
            components[i] = None
        else:
            components = OrderedDict()
            for c in self_.index_to_components(i):
                components[c] = None
        _enable_string_components(components, output)
        return output
    PatchInstanceMethod(function_space, "extract_sub_space", custom_extract_sub_space).patch()
    
def _preserve_root_space_after_sub(function_space, root_space_after_sub):
    function_space._root_space_after_sub = root_space_after_sub
    
    original_sub = function_space.sub
    def custom_sub(self_, i):
        output = original_sub(i)
        _preserve_root_space_after_sub(output, self_)
        return output
    PatchInstanceMethod(function_space, "sub", custom_sub).patch()
    
def _init_component_to_index(components, function_space):
    assert isinstance(components, (list, OrderedDict))
    if isinstance(components, list):
        function_space._component_to_index = OrderedDict()
        for (index, component) in enumerate(components):
            _init_component_to_index__recursive(component, function_space._component_to_index, index)
    else:
        function_space._component_to_index = components
    function_space._index_to_components = dict()
    for (component, index) in function_space._component_to_index.items():
        assert isinstance(index, (int, tuple)) or index is None
        if isinstance(index, int) or index is None:
            components = function_space._index_to_components.get(index, list())
            components.append(component)
            function_space._index_to_components[index] = components
        elif isinstance(index, tuple):
            for i in range(1, len(index) + 1):
                index_i = index[:i] if i > 1 else index[0]
                components = function_space._index_to_components.get(index_i, list())
                components.append(component)
                function_space._index_to_components[index_i] = components
        else:
            raise TypeError("Invalid index")
    def component_to_index(self_, i):
        return self_._component_to_index[i]
    AttachInstanceMethod(function_space, "component_to_index", component_to_index).attach()
    def index_to_components(self_, c):
        return self_._index_to_components[c]
    AttachInstanceMethod(function_space, "index_to_components", index_to_components).attach()
    
    original_collapse = function_space.collapse
    def custom_collapse(self_, collapsed_dofs=False):
        if not collapsed_dofs:
            output = original_collapse(collapsed_dofs)
        else:
            output, collapsed_dofs_dict = original_collapse(collapsed_dofs)
        if hasattr(self_, "_component_to_index"):
            _init_component_to_index(self_._component_to_index, output)
        if not collapsed_dofs:
            return output
        else:
            return output, collapsed_dofs_dict
    PatchInstanceMethod(function_space, "collapse", custom_collapse).patch()
    
def _init_component_to_index__recursive(components, component_to_index, index):
    assert isinstance(components, (str, tuple, list))
    if isinstance(components, str):
        if isinstance(index, list):
            component_to_index[components] = tuple(index)
        else:
            assert isinstance(index, int)
            component_to_index[components] = index
    elif isinstance(components, list):
        for component in components:
            _init_component_to_index__recursive(component, component_to_index, index)
    elif isinstance(components, tuple):
        for (subindex, subcomponent) in enumerate(components):
            full_index = list()
            if isinstance(index, int):
                full_index.append(index)
            else:
                full_index.extend(index)
            full_index.append(subindex)
            _init_component_to_index__recursive(subcomponent, component_to_index, full_index)
            
def _convert_component_to_int(function_space, i):
    if isinstance(i, str):
        return function_space._component_to_index[i]
    else:
        return i
