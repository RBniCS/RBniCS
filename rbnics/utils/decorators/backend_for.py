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

import inspect
import types
from rbnics.utils.decorators.dispatch import dispatch, Dispatcher

def BackendFor(library, inputs=None, replaces=None, replaces_if=None):
    def BackendFor_Decorator(Class):
        assert inspect.isclass(Class)
        assert hasattr(_cache, Class.__name__) # it was either added by @AbstractBackend or by previous @BackendFor
        if not isinstance(getattr(_cache, Class.__name__), Dispatcher): # added by @AbstractBackend, first time @BackendFor is called for this class name
            delattr(_cache, Class.__name__) # make space for dispatcher object
        _cache.__all__.add(Class.__name__)
        return dispatch(*inputs, module=_cache, replaces=replaces, replaces_if=replaces_if)(Class)
    return BackendFor_Decorator
    
def backend_for(library, inputs=None, replaces=None, replaces_if=None):
    def backend_for_decorator(function):
        assert inspect.isfunction(function)
        assert hasattr(_cache, function.__name__) # it was either added by @abstract_backend or by previous @backend_for
        if not isinstance(getattr(_cache, function.__name__), Dispatcher): # added by @abstract_backend, first time @backend_for is called for this function name
            delattr(_cache, function.__name__) # make space for dispatcher object
        _cache.__all__.add(function.__name__)
        return dispatch(*inputs, module=_cache, replaces=replaces, replaces_if=replaces_if)(function)
    return backend_for_decorator
    
_cache = types.ModuleType("backends", "Storage for backends") # cannot import directly in rbnics.backends because of circular imports
_cache.__all__ = set()
