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
## @file truth_vector.py
#  @brief Type of truth vector
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

# Declare abstract vector type
from abc import ABCMeta, abstractmethod
import inspect
import itertools
from functools import wraps
from RBniCS.utils.mpi import log, DEBUG

def BackendFor(library, online_backend=None, inputs=None):
    def BackendFor_Decorator(Class):
        log(DEBUG,
            "In BackendFor with\n" +
            "\tlibrary = " + str(library) + "\n" +
            "\tonline_backend = " + str(online_backend) + "\n" +
            "\tinputs = " + str(inputs) + "\n" +
            "\tClass = " + str(Class) + "\n"
        )
        
        assert inspect.isclass(Class)
        
        if not library in BackendFor._all_classes:
            BackendFor._all_classes[library] = dict() # from class name to class (if no online backend) or dict over online backends (if online backed provided)
            
        if online_backend is None:
            assert Class.__name__ not in BackendFor._all_classes[library]
            BackendFor._all_classes[library][Class.__name__] = Class
        else:
            if Class.__name__ not in BackendFor._all_classes[library]:
                BackendFor._all_classes[library][Class.__name__] = dict() # from online_backend to class
            BackendFor._all_classes[library][Class.__name__][online_backend] = Class
        
        if library is not "Abstract":
            # TODO check that the signature are the same, or at worst there have been added arguments with default values (or new methods, not required by the interface)
            
            assert not hasattr(Class, "Type") # this is used only by functions
        
            assert inputs is not None
            assert isinstance(inputs, tuple)
            validate_inputs(inputs)
            if Class.__name__ not in BackendFor._all_classes_inputs:
                BackendFor._all_classes_inputs[Class.__name__] = dict() # from inputs to library
            for possible_inputs in combine_inputs(inputs):
                assert possible_inputs not in BackendFor._all_classes_inputs[Class.__name__], "Input types " + str(possible_inputs) + " for " + Class.__name__ + " cannot be the same for backends " + BackendFor._all_classes_inputs[Class.__name__][possible_inputs] + " (already in storage) and " + library + " (to be added), otherwise it will not be possible to choose among them"
                BackendFor._all_classes_inputs[Class.__name__][possible_inputs] = library
        
        log(DEBUG,
            "After BackendFor, backends classes storage contains\n" +
            "\tclasses dict: " + logging_all_classes_functions(BackendFor._all_classes) + "\n" +
            "\tclasses inputs dict: " + logging_all_classes_functions_inputs(BackendFor._all_classes_inputs) + "\n"
        )
        
        return Class
    return BackendFor_Decorator
    
BackendFor._all_classes = dict() # from library to dict from class name to class
BackendFor._all_classes_inputs = dict() # from inputs to library
    
def backend_for(library, online_backend=None, inputs=None, output=None):
    def backend_for_decorator(function):
        log(DEBUG,
            "In backend_for with\n" +
            "\tlibrary = " + str(library) +
            "\tonline_backend = " + str(online_backend) +
            "\tinputs = " + str(inputs) +
            "\tfunction = " + str(function)
        )
        
        assert inspect.isfunction(function)
        
        if not library in backend_for._all_functions:
            backend_for._all_functions[library] = dict() # from function name to function (if no online backend) or dict over online backends (if online backed provided)
        
        if online_backend is None:
            assert function.__name__ not in backend_for._all_functions[library]
            backend_for._all_functions[library][function.__name__] = function
        else:
            if function.__name__ not in backend_for._all_functions[library]:
                backend_for._all_functions[library][function.__name__] = dict() # from online_backend to function
            backend_for._all_functions[library][function.__name__][online_backend] = function
            
        if library is not "Abstract":
            # TODO check that the signature are the same, or at worst there have been added arguments with default values
            
            if output is not None:
                def Type():
                    return output
                
                function.Type = Type
                            
            assert inputs is not None
            assert isinstance(inputs, tuple)
            validate_inputs(inputs)
            if function.__name__ not in backend_for._all_functions_inputs:
                backend_for._all_functions_inputs[function.__name__] = dict() # from inputs to library
            for possible_inputs in combine_inputs(inputs):
                assert possible_inputs not in backend_for._all_functions_inputs[function.__name__], "Input types " + str(possible_inputs) + " for " + function.__name__ + " cannot be the same for backends " + backend_for._all_functions_inputs[function.__name__][possible_inputs] + " (already in storage) and " + library + " (to be added), otherwise it will not be possible to choose among them"
                backend_for._all_functions_inputs[function.__name__][possible_inputs] = library
            
        
        log(DEBUG,
            "After backend_for, backends function storage contains\n" +
            "\tfunction dict: " + logging_all_classes_functions(backend_for._all_functions) + "\n" +
            "\tfunction inputs dict: " + logging_all_classes_functions_inputs(backend_for._all_functions_inputs) + "\n"
        )
        
        return function
    return backend_for_decorator

backend_for._all_functions = dict() # from library to dict from function name to function
backend_for._all_functions_inputs = dict() # from inputs to library

def combine_inputs(inputs):
    # itertools.product does not work with types if there is only one element.
    converted_inputs = list()
    for i in inputs:
        if isinstance(i, tuple):
            converted_inputs.append(i)
        else:
            converted_inputs.append((i, ))
    return itertools.product(*converted_inputs)
    
def validate_inputs(inputs):
    for input_ in inputs:
        if isinstance(input_, (list, tuple)):
            validate_inputs(input_)
        else:
            assert input_ is not list, "Please use list_of defined in this module to specify the type of each element"
            assert input_ is not tuple, "Please use tuple_of defined in this module to specify the type of each element"
            assert inspect.isclass(input_) or isinstance(input_, (list_of, tuple_of)) or input_ is None
    
def logging_all_classes_functions(storage):
    output = "{" + "\n"
    for library in storage:
        output += "\t\t" + library + ": {" + "\n"
        for name in storage[library]:
            if isinstance(storage[library][name], dict):
                output += "\t\t\t" + name + ": {" + "\n"
                for online_backend in storage[library][name]:
                    output += "\t\t\t\t" + online_backend + ": " + str(storage[library][name][online_backend]) + "\n"
                output += "\t\t\t" + "}" + "\n"
            else:
                output += "\t\t\t" + name + ": " + str(storage[library][name]) + "\n"
        output += "\t\t" + "}" + "\n"
    output += "\t" + "}"
    return output
    
def logging_all_classes_functions_inputs(storage):
    output = "{" + "\n"
    for name in storage:
        output += "\t\t" + name + ": {" + "\n"
        for inputs in storage[name]:
            output += "\t\t\t" + str(inputs) + ": " + str(storage[name][inputs]) + "\n"
        output += "\t\t" + "}" + "\n"
    output += "\t" + "}"
    return output
    
# Helper functions to be more precise when input types are tuple or list
class _tuple_or_list_of(object):
    def __init__(self, types):
        self.types = types
        
    def are_subclass(self, other):
        def is_subclass(item_self, item_other):
            if isinstance(item_self, _tuple_or_list_of) and isinstance(item_other, _tuple_or_list_of):
                return item_self.are_subclass(item_other)
            elif isinstance(item_self, _tuple_or_list_of) or isinstance(item_other, _tuple_or_list_of): # but not both
                return False
            else:
                return issubclass(item_self, item_other)
                
        if not isinstance(other, _tuple_or_list_of) or type(self) != type(other):
            return False
        elif isinstance(self.types, tuple) and isinstance(other.types, tuple):
            if len(self.types) != len(other.types):
                return False
            else:
                for (item_self, item_other) in zip(sorted(self.types), sorted(other.types)):
                    if not is_subclass(item_self, item_other):
                        return False
                return True
        elif isinstance(self.types, tuple) or isinstance(other.types, tuple): # but not both
            return False
        else:
            assert not isinstance(self.types, list), "Please use tuples instead"
            return is_subclass(self.types, other.types)

class tuple_of(_tuple_or_list_of):        
    def __str__(self):
        return "tuple_of(" + str(self.types) + ")"
    __repr__ = __str__
        
class list_of(_tuple_or_list_of):
    def __str__(self):
        return "list_of(" + str(self.types) + ")"
    __repr__ = __str__

from numpy import float64
ThetaType = (tuple_of(float), tuple_of(float64),  tuple_of(int), tuple_of((float, float64)), tuple_of((float, int)), tuple_of((float64, int)), tuple_of((float, float64, int)))
