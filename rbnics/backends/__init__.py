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

import importlib
import os
import sys
current_module = sys.modules[__name__]
current_directory = os.path.dirname(os.path.realpath(__file__))
available_backends = list()
for root, dirs, files in os.walk(current_directory):
    available_backends.extend(dirs)
    break # prevent recursive exploration
available_backends.remove("abstract")
available_backends.remove("basic")
available_backends.remove("common")
available_backends.remove("online")

# Make sure to import all available backends, so that they are added to the factory storage
import rbnics.backends.abstract
import rbnics.backends.common
for backend in available_backends:
    importlib.import_module("rbnics.backends." + backend)
    importlib.import_module("rbnics.backends." + backend + ".wrapping")

# Combine all enabled backends available in the factory and store them in this module
from rbnics.utils.factories import backends_factory, enable_backend
enable_backend("common")
for backend in available_backends:
    enable_backend(backend)
backends_factory(current_module)

# Store some additional classes, defined in the abstract module, which are base classes but not backends, 
# and thus have not been processed by the enabled_backend function above
from rbnics.backends.abstract import LinearProblemWrapper, NonlinearProblemWrapper, TimeDependentProblemWrapper, TimeDependentProblem1Wrapper, TimeDependentProblem2Wrapper

# Next, extend modules with __overridden__ variables in backends wrapping. In order to account for multiple overrides,
# sort the list of available backends to account that
depends_on_backends = dict()
at_least_one_dependent_backend = False
for backend in available_backends:
    if hasattr(sys.modules["rbnics.backends." + backend + ".wrapping"], "__depends_on__"):
        depends_on_backends[backend] = set(sys.modules["rbnics.backends." + backend + ".wrapping"].__depends_on__)
        at_least_one_dependent_backend = True
    else:
        depends_on_backends[backend] = set()
if at_least_one_dependent_backend:
    from toposort import toposort_flatten
    available_backends = toposort_flatten(depends_on_backends)

# Extend parent module __all__ variable with backends wrapping
for backend in available_backends:
    if hasattr(sys.modules["rbnics.backends." + backend + ".wrapping"], "__overridden__"):
        wrapping_overridden = sys.modules["rbnics.backends." + backend + ".wrapping"].__overridden__
        assert isinstance(wrapping_overridden, dict)
        for (module_name, classes_or_functions) in wrapping_overridden.iteritems():
            assert isinstance(classes_or_functions, (list, dict))
            if isinstance(classes_or_functions, list):
                classes_or_functions = dict((class_or_function, class_or_function) for class_or_function in classes_or_functions)
            for (class_or_function_name, class_or_function_impl) in classes_or_functions.iteritems():
                setattr(sys.modules[module_name], class_or_function_name, getattr(sys.modules["rbnics.backends." + backend + ".wrapping"], class_or_function_impl))
                if class_or_function_name not in sys.modules[module_name].__all__:
                    sys.modules[module_name].__all__.append(class_or_function_name)

# Clean up
del current_module
del current_directory
del available_backends
