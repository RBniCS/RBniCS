# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file reduced_problem_factory.py
#  @brief Factory to generate a reduced problem corresponding to a given reduction method and truth problem
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

#~~~~~~~~~~~~~~~~~~~~~~~~~     PARAMETRIZED PROBLEM BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ReducedProblemFactory
#

import inspect
from RBniCS.utils.decorators import BackendFor, backend_for
from RBniCS.utils.mpi import log, DEBUG

# Factory to combine all available backends
def backends_factory(backends_module):
    # Logging
    log(DEBUG,
        "In backends_factory with\n" +
        "\tbackends module = " + str(backends_module)
    )
    
    # Function to add attribute to backends module
    def add_backends_module_attr(input_map, return_map):
        for r in return_map["Abstract"]:
            def backend_selector(*args, **kwargs):
                input_types = list()
                input_types.extend([type(arg) for arg in args])
                input_types.extend([type(kwargs[key]) for key in kwargs])
                input_types = tuple(input_types)
                for i in input_map:
                    if i is input_types:
                        if input_map[i] in backends_factory._enabled_backends:
                            returned_class_or_function = return_map[ input_map[i] ][r]
                            if isinstance(ReturnType, dict):
                                return returned_class_or_function[online_backend_factory._online_backend](*args, **kwargs)
                            else:
                                return returned_class_or_function(*args, **kwargs)
                return TypeError("No backend found for return type " + r + " with input arguments " + input_types)
            setattr(backends_module, r, backend_selector)
    
    # Generate backend classes
    add_backends_module_attr(BackendFor._all_classes_inputs, BackendFor._all_classes)
    
    # Generate backend functions
    add_backends_module_attr(backend_for._all_functions_inputs, backend_for._all_functions)
    
    # Logging
    log(DEBUG,
        "After backends_factory, backends module contains\n" +
        "\t" + str(backends_module)
    )
    
backends_factory._enabled_backends = list() # of strings

# Enable or disable backends
def enable_backend(library):
    backends_factory._enabled_backends.append(library)
    
def disable_backend(library):
    assert library in backends_factory._enabled_backends
    backends_factory._enabled_backends.remove(library)
    
# Generate Online* classes
def online_backend_factory(backends_online_module):
    # Logging
    log(DEBUG,
        "In set_online_backend with\n" +
        "\tonline backend = " + online_backend_factory._online_backend +
        "\tbackends module = " + str(backends_online_module)
    )
    
    assert online_backend_factory._online_backend is not None
    
    # Function to add attribute to backends.online module
    def add_backends_module_attr(required, return_map):
        for item in required:
            def backend_selector(*args, **kwargs):
                returned_class_or_function = return_map[online_backend_factory._online_backend][item]
                if hasattr(returned_class_or_function, "Type"):
                    assert inspect.isfunction(returned_class_or_function) # we use it to store the output type of a function
                                                                          # if it were a class we would not need it
                    backend_selector.Type =  returned_class_or_function.Type
                return returned_class_or_function(*args, **kwargs)
            setattr(backends_online_module, "Online" + item, backend_selector)

    # Generate backend.online classes
    required_classes = [
        "AffineExpansionStorage", # to store the affine expansion in reduced problems
        "EigenSolver" # required by POD
    ]
    add_backends_module_attr(required_classes, BackendFor._all_classes)
    
    # Generate backend.online functions
    required_functions = [
        "Function", # e.g. for storage of the online solution
        "Matrix", "Vector" # e.g. to compute reduced data structures from high fidelity ones
    ]
    add_backends_module_attr(required_functions, backend_for._all_functions)
    
    # Logging
    log(DEBUG,
        "After set_online_backend, backends module contains\n" +
        "\t" + str(backends_online_module)
    )
    
online_backend_factory._online_backend = None

# Set the (unique) online backend
def set_online_backend(library):
    online_backend_factory._online_backend = library

