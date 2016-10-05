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
from RBniCS.utils.decorators import BackendFor, backend_for, list_of, tuple_of
from RBniCS.utils.decorators.backend_for import _list_of, _tuple_of
from RBniCS.utils.mpi import log, DEBUG

# Factory to combine all available backends
def backends_factory(backends_module):
    assert not backends_factory._already_called, "You do not need to call backends_factory again after enabling or disabling modules"
    backends_factory._already_called = True
    
    # Logging
    log(DEBUG,
        "In backends_factory with\n" +
        "\tbackends module = " + str(backends_module)
    )
    
    # Function to add attribute to backends module
    def add_backends_module_attr(input_map, return_map):
        for class_or_function_name in return_map["Abstract"]:
            log(DEBUG, "\t\tAdding backend selector for " + class_or_function_name)
            backend_selector = create_backend_selector(class_or_function_name, input_map, return_map)
            setattr(backend_selector, "__name__", class_or_function_name)
            setattr(backend_selector, "__module__", backends_module)
            setattr(backends_module, class_or_function_name, backend_selector)
    
    # Function to create a backends selector
    def create_backend_selector(class_or_function_name, input_map, return_map):
        def backend_selector(*args, **kwargs):
            inputs = list()
            inputs.extend([arg for arg in args])
            inputs.extend([kwargs[key] for key in kwargs])
            if len(inputs) > 0:
                input_types = get_input_types(inputs)
                log(DEBUG,
                    "In backend_selector with\n" +
                    "\trequested class or function = " + str(class_or_function_name) + "\n" +
                    "\tprovided inputs = " + str(input_types)
                )
                for (backend_input_types, corresponding_backend) in input_map[class_or_function_name].iteritems():
                    if are_subclass(input_types, backend_input_types):
                        if corresponding_backend in backends_factory._enabled_backends:
                            returned_class_or_function = return_map[corresponding_backend][class_or_function_name]
                            if isinstance(returned_class_or_function, dict):
                                returned_class_or_function = returned_class_or_function[online_backend_factory._online_backend]
                            log(DEBUG,
                                "\tcorresponding backend = " + corresponding_backend + "\n" +
                                "\tcorresponding backend class or function = " + str(returned_class_or_function) + "\n"
                            )
                            return returned_class_or_function(*args, **kwargs)
                else:
                    error_message = "No backend found for return type " + str(class_or_function_name) + " with input arguments " + str(input_types) + ".\n"
                    error_message += "Available input types for " + str(class_or_function_name) + " are:\n"
                    for (backend_input_types, corresponding_backend) in input_map[class_or_function_name].iteritems():
                        error_message += "\t" + str(backend_input_types) + ": " + corresponding_backend + "\n"
                    error_message += "\n"
                    raise TypeError(error_message)
            else: # used in some constructors
                log(DEBUG,
                    "In backend_selector with\n" +
                    "\trequested class or function = " + str(class_or_function_name) + "\n" +
                    "\tprovided inputs = " + str(None)  + "\n" +
                    "\tcorresponding backend = " + str(None) + "\n" +
                    "\tcorresponding backend class or function = " + str(None) + "\n"
                )
                return None
        return backend_selector
    
    # Function to get input types
    def get_input_types(inputs):
        input_types = list()
        for input_ in inputs:
            if type(input_) in (list, tuple): # more strict that isinstance(input_, (list, tuple)): custom types inherited from list or tuple should be preserved
                input_subtypes = get_input_types(input_)
                input_subtypes = tuple(set(input_subtypes)) # remove repeated types
                if len(input_subtypes) == 1:
                    input_subtypes = input_subtypes[0]
                if isinstance(input_, list):
                    input_types.append(list_of(input_subtypes))
                elif isinstance(input_, tuple):
                    input_types.append(tuple_of(input_subtypes))
                else:
                    raise TypeError("Invalid type in get_input_types()")
            else:
                if input_ is not None:
                    input_types.append(type(input_))
        input_types = tuple(input_types)
        return input_types
    
    # Generate backend classes
    log(DEBUG, "\tGenerate backend classes")
    add_backends_module_attr(BackendFor._all_classes_inputs, BackendFor._all_classes)
    
    # Generate backend functions
    log(DEBUG, "\tGenerate backend functions")
    add_backends_module_attr(backend_for._all_functions_inputs, backend_for._all_functions)
    
    # Logging
    log(DEBUG,
        "After backends_factory, backends module contains\n" +
        logging_backends_module(backends_module)
    )
    
backends_factory._enabled_backends = list() # of strings
backends_factory._already_called = False

# Enable or disable backends
def enable_backend(library):
    log(DEBUG, "Enabling backend " + library)
    backends_factory._enabled_backends.append(library)
    
def disable_backend(library):
    log(DEBUG, "Disabling backend " + library)
    assert library in backends_factory._enabled_backends
    backends_factory._enabled_backends.remove(library)
    
# Generate Online* classes
def online_backend_factory(backends_online_module):
    # Logging
    log(DEBUG,
        "In online_backend_factory with\n" +
        "\tonline backend = " + online_backend_factory._online_backend + "\n" +
        "\tbackends module = " + str(backends_online_module)
    )
    
    assert online_backend_factory._online_backend is not None
    
    # Function to add attribute to backends.online module
    def add_backends_module_attr(return_map):
        for class_or_function_name in return_map["Abstract"]:
            log(DEBUG, "\t\tAdding online backend selector for " + class_or_function_name)
            backend_selector = create_backend_selector(class_or_function_name, return_map)
            setattr(backend_selector, "__name__", class_or_function_name)
            setattr(backend_selector, "__module__", backends_online_module)
            if class_or_function_name[0].isupper():
                assert "_" not in class_or_function_name
                setattr(backends_online_module, "Online" + class_or_function_name, backend_selector)
            else:
                setattr(backends_online_module, "online_" + class_or_function_name, backend_selector)
            
    # Function to create a backends selector
    def create_backend_selector(class_or_function_name, return_map):
        def backend_selector_returned_class_or_function():
            returned_class_or_function = return_map[online_backend_factory._online_backend][class_or_function_name]
            if isinstance(returned_class_or_function, dict):
                returned_class_or_function = returned_class_or_function[online_backend_factory._online_backend]
            return returned_class_or_function
        def backend_selector(*args, **kwargs):
            if len(args) + len(kwargs) > 0:
                returned_class_or_function = backend_selector_returned_class_or_function()
                return returned_class_or_function(*args, **kwargs)
            else: # used in some constructors
                return None
        if hasattr(return_map["Abstract"][class_or_function_name], "Type"):
            def backend_selector_Type():
                returned_class_or_function = backend_selector_returned_class_or_function()
                return returned_class_or_function.Type()
            backend_selector.Type = backend_selector_Type
        return backend_selector

    # Generate backend.online classes
    log(DEBUG, "\tGenerate online backend classes")
    add_backends_module_attr(BackendFor._all_classes)
    
    # Generate backend.online functions
    log(DEBUG, "\tGenerate online backend functions")
    add_backends_module_attr(backend_for._all_functions)
    
    # Logging
    log(DEBUG,
        "After online_backend_factory, backends module contains\n" +
        logging_online_backends_module(backends_online_module)
    )
    
online_backend_factory._online_backend = None

# Set the (unique) online backend
def set_online_backend(library):
    log(DEBUG, "Set online backend to " + library)
    online_backend_factory._online_backend = library

# Helper
def logging_backends_module(module):
    output = ""
    for attribute_name in dir(module):
        if not attribute_name.startswith("__"): # do not show standard python attributes
            attribute_function = getattr(module, attribute_name)
            if inspect.isfunction(attribute_function):
                output += "\t" + attribute_name + ": " + str(attribute_function) + "\n"
    return output
    
def logging_online_backends_module(module):
    output = ""
    for attribute_name in dir(module):
        if not attribute_name.startswith("__"): # do not show standard python attributes
            attribute_function = getattr(module, attribute_name)
            if inspect.isfunction(attribute_function):
                output += "\t" + attribute_name + ": " + str(attribute_function)
                if hasattr(attribute_function, "Type"):
                    output += " (Type: " + str(attribute_function.Type) + ")"
                output += "\n"
    return output
    
def are_subclass(input_types, backend_input_types):
    assert isinstance(input_types, tuple)
    assert isinstance(backend_input_types, tuple)
    backend_input_types = list(backend_input_types)
    if None in backend_input_types:
        backend_input_types.remove(None) # strip default None argument from backend input types
    if len(input_types) != len(backend_input_types):
        return False
    else:
        for (input_type, backend_input_type) in zip(input_types, backend_input_types):
            if isinstance(input_type, _tuple_of):
                if isinstance(backend_input_type, _tuple_of):
                    if not input_type.are_subclass(backend_input_type):
                        return False
                else:
                    return False
            elif isinstance(input_type, _list_of):
                if isinstance(backend_input_type, _list_of):
                    if not input_type.are_subclass(backend_input_type):
                        return False
                else:
                    return False
            else:
                if isinstance(backend_input_type, (_tuple_of, _list_of)):
                    return False
                elif not issubclass(input_type, backend_input_type):
                    return False
        else:
            return True
            
