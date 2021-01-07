# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import inspect
import types
from rbnics.utils.decorators.dispatch import dispatch, Dispatcher


def BackendFor(library, inputs=None, replaces=None, replaces_if=None):
    def BackendFor_Decorator(Class):
        assert inspect.isclass(Class)
        assert hasattr(_cache, Class.__name__)  # it was either added by @AbstractBackend or previous @BackendFor
        if not isinstance(getattr(_cache, Class.__name__), Dispatcher):
            # added by @AbstractBackend, first time @BackendFor is called for this class name
            delattr(_cache, Class.__name__)  # make space for dispatcher object
        _cache.__all__.add(Class.__name__)
        return dispatch(*inputs, module=_cache, replaces=replaces, replaces_if=replaces_if)(Class)
    return BackendFor_Decorator


def backend_for(library, inputs=None, replaces=None, replaces_if=None):
    def backend_for_decorator(function):
        assert inspect.isfunction(function)
        assert hasattr(_cache, function.__name__)  # it was either added by @abstract_backend or previous @backend_for
        if not isinstance(getattr(_cache, function.__name__), Dispatcher):
            # added by @abstract_backend, first time @backend_for is called for this function name
            delattr(_cache, function.__name__)  # make space for dispatcher object
        _cache.__all__.add(function.__name__)
        return dispatch(*inputs, module=_cache, replaces=replaces, replaces_if=replaces_if)(function)
    return backend_for_decorator


_cache = types.ModuleType("backends", "Storage for backends")
# cannot import directly in rbnics.backends because of circular imports
_cache.__all__ = set()
