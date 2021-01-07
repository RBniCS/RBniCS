# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Declare abstract vector type
import inspect
from functools import wraps
from rbnics.utils.decorators.backend_for import _cache as backends_cache


def AbstractBackend(Class):
    assert inspect.isclass(Class)
    assert hasattr(Class, "__abstractmethods__")  # this means that ABCMeta was used as metaclass, see PEP 3119

    assert not hasattr(backends_cache, Class.__name__)
    setattr(backends_cache, Class.__name__, Class)
    backends_cache.__all__.add(Class.__name__)
    return Class


def abstract_backend(function):
    assert inspect.isfunction(function)

    @wraps(function)
    def abstract_backend_function(*args, **kwargs):
        raise NotImplementedError("This function is just a placeholder, it should never get called."
                                  + " If you see this error you have probably forgotten to implement"
                                  + " a function in your backend.")

    assert not hasattr(backends_cache, function.__name__)
    setattr(backends_cache, function.__name__, abstract_backend_function)
    backends_cache.__all__.add(function.__name__)
    return abstract_backend_function


def abstract_online_backend(function):
    assert inspect.isfunction(function)

    @wraps(function)
    def abstract_online_backend_function(*args, **kwargs):
        raise NotImplementedError("This function is just a placeholder, it should never get called."
                                  + " If you see this error you have probably forgotten to implement"
                                  + " a function in your backend, or you are trying to call this method"
                                  + " with a backend which is not supposed to be used online.")

    assert not hasattr(backends_cache, function.__name__)
    setattr(backends_cache, function.__name__, abstract_online_backend_function)
    backends_cache.__all__.add(function.__name__)
    return abstract_online_backend_function


def abstractonlinemethod(method):
    assert inspect.isfunction(method)

    @wraps(method)
    def abstractonlinemethod_function(*args, **kwargs):
        raise NotImplementedError("This method is just a placeholder, it should never get called."
                                  + " If you see this error you have probably forgotten to implement"
                                  + " a method in your backend, or you are trying to call this method"
                                  + " with a backend which is not supposed to be used online.")

    return abstractonlinemethod_function
