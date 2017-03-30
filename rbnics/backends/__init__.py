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
## @file __init__.py
#  @brief Init file for auxiliary linear algebra module
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

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

# Combine all enabled backends available in the factory and store them in this module
from rbnics.utils.factories import backends_factory, enable_backend
enable_backend("common")
for backend in available_backends:
    enable_backend(backend)
backends_factory(current_module)

# Extend parent module __all__ variable with backends wrapping
for backend in available_backends:
    importlib.import_module("rbnics.backends." + backend + ".wrapping")
    wrapping_overridden = sys.modules["rbnics.backends." + backend + ".wrapping"].__overridden__
    for class_or_function in wrapping_overridden:
        assert not hasattr(sys.modules["rbnics"], class_or_function)
        setattr(sys.modules["rbnics"], class_or_function, getattr(sys.modules["rbnics.backends." + backend + ".wrapping"], class_or_function))
        sys.modules["rbnics"].__all__ += [class_or_function]

# Clean up
del current_module
del current_directory
del available_backends
