# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import importlib
import sys

# Initialize __all__ variable
__all__ = []

# Process configuration files first
from rbnics.utils.config import config


# Helper function to load required backends
def load_backends(required_backends):
    # Clean up backends cache
    from rbnics.utils.decorators.backend_for import _cache as backends_cache
    for class_or_function_name in backends_cache.__all__:
        delattr(backends_cache, class_or_function_name)
        if hasattr(sys.modules[__name__], class_or_function_name):
            delattr(sys.modules[__name__], class_or_function_name)
            assert class_or_function_name in sys.modules[__name__].__all__
            sys.modules[__name__].__all__.remove(class_or_function_name)
    backends_cache.__all__ = set()

    # Make sure to import all available backends, so that they are added to the backends cache
    importlib.import_module(__name__ + ".abstract")
    importlib.import_module(__name__ + ".common")
    for backend in required_backends:
        importlib.import_module(__name__ + "." + backend)
        importlib.import_module(__name__ + "." + backend + ".wrapping")
    importlib.import_module(__name__ + ".online")

    # Copy imported backends from backends cache to this module
    for class_or_function_name in backends_cache.__all__:
        assert not hasattr(sys.modules[__name__], class_or_function_name)
        setattr(sys.modules[__name__], class_or_function_name, getattr(backends_cache, class_or_function_name))
        sys.modules[__name__].__all__.append(class_or_function_name)

    # Next, extend modules with __overridden__ variables in backends wrapping. In order to account for
    # multiple overrides, sort the list of available backends to account that
    depends_on_backends = dict()
    at_least_one_dependent_backend = False
    for backend in required_backends:
        if hasattr(sys.modules[__name__ + "." + backend + ".wrapping"], "__depends_on__"):
            depends_on_backends[backend] = set(sys.modules[__name__ + "." + backend + ".wrapping"].__depends_on__)
            at_least_one_dependent_backend = True
        else:
            depends_on_backends[backend] = set()
    if at_least_one_dependent_backend:
        from toposort import toposort_flatten
        required_backends = toposort_flatten(depends_on_backends)

    # Apply possible overriddes defined in backends wrapping
    for backend in required_backends:
        if hasattr(sys.modules[__name__ + "." + backend + ".wrapping"], "__overridden__"):
            wrapping_overridden = sys.modules[__name__ + "." + backend + ".wrapping"].__overridden__
            assert isinstance(wrapping_overridden, dict)
            for (module_name, classes_or_functions) in wrapping_overridden.items():
                assert isinstance(classes_or_functions, (list, dict))
                if isinstance(classes_or_functions, list):
                    classes_or_functions = dict((class_or_function, class_or_function)
                                                for class_or_function in classes_or_functions)
                for (class_or_function_name, class_or_function_impl) in classes_or_functions.items():
                    setattr(sys.modules[module_name], class_or_function_name,
                            getattr(sys.modules[__name__ + "." + backend + ".wrapping"], class_or_function_impl))
                    if hasattr(sys.modules[module_name], "__all__"):
                        if class_or_function_name not in sys.modules[module_name].__all__:
                            sys.modules[module_name].__all__.append(class_or_function_name)

    # Make sure that backends do not contain dispatcher functions (but rather, actual functions and classes)
    import inspect
    for backend in required_backends:
        for class_or_function_name in sys.modules[__name__ + "." + backend].__all__:
            class_or_function = getattr(sys.modules[__name__ + "." + backend], class_or_function_name)
            assert inspect.isclass(class_or_function) or inspect.isfunction(class_or_function)

    # In contrast, make sure that this module only contains dispatcher objects
    from rbnics.utils.decorators.dispatch import Dispatcher
    for dispatcher_name in sys.modules[__name__].__all__:
        dispatcher = getattr(sys.modules[__name__], dispatcher_name)
        # if there was at least a concrete implementation by @BackendFor or @backend_for
        if isinstance(getattr(backends_cache, dispatcher_name), Dispatcher):
            assert isinstance(dispatcher, Dispatcher)

    # Store some additional classes, defined in the abstract module, which are base classes but not backends,
    # and thus have not been processed by @BackendFor and @backend_for decorators
    for extra_class in ("LinearProblemWrapper", "NonlinearProblemWrapper", "TimeDependentProblemWrapper"):
        assert not hasattr(sys.modules[__name__], extra_class)
        setattr(sys.modules[__name__], extra_class, getattr(sys.modules[__name__ + ".abstract"], extra_class))
        sys.modules[__name__].__all__.append(extra_class)


# Load required backends
load_backends(config.get("backends", "required backends"))
