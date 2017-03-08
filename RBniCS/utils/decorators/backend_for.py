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
from numpy import ndarray as array, float64
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
            assert online_backend not in BackendFor._all_classes[library][Class.__name__]
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
    
def SameBackendFor(library, source_library, Class, online_backend=None, inputs=None):
    # TODO check that inputs are the same for library and source_library
    pass
    
def OverrideBackendFor(library, online_backend=None, inputs=None):
    def OverrideBackendFor_Decorator(Class):
        if online_backend is None:
            assert Class.__name__ in BackendFor._all_classes[library]
            del BackendFor._all_classes[library][Class.__name__]
        else:
            assert online_backend in BackendFor._all_classes[library][Class.__name__]
            del BackendFor._all_classes[library][Class.__name__][online_backend]
            
        if library is not "Abstract":
            for possible_inputs in combine_inputs(inputs):
                assert possible_inputs in BackendFor._all_classes_inputs[Class.__name__]
                del BackendFor._all_classes_inputs[Class.__name__][possible_inputs]
            
        return BackendFor(library, online_backend, inputs)(Class)
    return OverrideBackendFor_Decorator
    
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
            assert online_backend not in backend_for._all_functions[library][function.__name__]
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
    
def same_backend_for(library, source_library, Class, online_backend=None, inputs=None, output=None):
    # TODO check that inputs/output are the same for library and source_library
    pass
    
def override_backend_for(library, online_backend=None, inputs=None, output=None):
    def override_backend_for_decorator(function):
        if online_backend is None:
            assert function.__name__ in backend_for._all_functions[library]
            del backend_for._all_functions[library][function.__name__]
        else:
            assert online_backend in backend_for._all_functions[library][function.__name__]
            del backend_for._all_functions[library][function.__name__][online_backend]
            
        if library is not "Abstract":
            for possible_inputs in combine_inputs(inputs):
                assert possible_inputs in backend_for._all_functions_inputs[function.__name__]
                del backend_for._all_functions_inputs[function.__name__][possible_inputs]
                
        return backend_for(library, online_backend, inputs, output)(function)
    return override_backend_for_decorator

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
        if (
            type(input_) in (list, tuple) # more strict than isinstance(input_, (list, tuple)): custom types inherited from array or list or tuple should be preserved
                or
            (type(input_) in (array, ) and input_.dtype == object)
        ):
            validate_inputs(input_)
        else:
            assert not (input_ is array and input_.dtype == object), "Please use array_of defined in this module to specify the type of each element"
            assert input_ is not dict, "Please use dict_of defined in this module to specify the type of keys and values"
            assert input_ is not list, "Please use list_of defined in this module to specify the type of each element"
            assert input_ is not tuple, "Please use tuple_of defined in this module to specify the type of each element"
            assert inspect.isclass(input_) or isinstance(input_, (_array_of, _dict_of, _list_of, _tuple_of)) or input_ is None
    
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
def _tuple_or_list_or_array_of__collapse(types, ChildClass):
    if not ChildClass is None and isinstance(types, tuple) and all([isinstance(t, ChildClass) for t in types]):
        all_types = set()
        for t in types:
            if isinstance(t.types, tuple):
                for tt in t.types:
                    assert inspect.isclass(tt)
                all_types.update(t.types)
            else:
                assert inspect.isclass(t.types)
                all_types.add(t.types)
        return tuple(all_types)
    else:
        return types
        
class _tuple_or_list_or_array_of(object):
    def __init__(self, types):
        self.types = types
        
    def are_subclass(self, other):
        def is_subclass(item_self, item_other):
            if isinstance(item_self, _tuple_or_list_or_array_of) and isinstance(item_other, _tuple_or_list_or_array_of):
                return item_self.are_subclass(item_other)
            elif isinstance(item_self, _tuple_or_list_or_array_of) or isinstance(item_other, _tuple_or_list_or_array_of): # but not both
                return False
            else:
                return issubclass(item_self, item_other)
                
        if not isinstance(other, _tuple_or_list_or_array_of) or type(self) != type(other):
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

class _tuple_of(_tuple_or_list_or_array_of):
    def __str__(self):
        return "tuple_of(" + str(self.types) + ")"
    __repr__ = __str__
    
class _list_of(_tuple_or_list_or_array_of):
    def __str__(self):
        return "list_of(" + str(self.types) + ")"
    __repr__ = __str__
    
class _array_of(_tuple_or_list_or_array_of):
    def __str__(self):
        return "array_of(" + str(self.types) + ")"
    __repr__ = __str__
    
class _dict_of(object):
    def __init__(self, types_from, types_to):
        self.types_from = _tuple_or_list_or_array_of(types_from)
        self.types_to = _tuple_or_list_or_array_of(types_to)
        
    def are_subclass(self, other):
        return self.types_from.are_subclass(other.types_from) and self.types_to.are_subclass(other.types_to)
        
    def __str__(self):
        return "dict_of(" + str(self.types_from.types) + ": " + str(self.types_to.types) + ")"
    __repr__ = __str__
    
_all_tuple_of_instances = dict()
_all_list_of_instances = dict()
_all_array_of_instances = dict()
_all_dict_of_instances = dict()

def tuple_of(types):
    types = _tuple_or_list_or_array_of__collapse(types, _tuple_of)
    if types not in _all_tuple_of_instances:
        _all_tuple_of_instances[types] = _tuple_of(types)
    return _all_tuple_of_instances[types]
    
def list_of(types):
    types = _tuple_or_list_or_array_of__collapse(types, _list_of)
    if types not in _all_list_of_instances:
        _all_list_of_instances[types] = _list_of(types)
    return _all_list_of_instances[types]
    
def array_of(types):
    types = _tuple_or_list_or_array_of__collapse(types, _array_of)
    if types not in _all_array_of_instances:
        _all_array_of_instances[types] = _array_of(types)
    return _all_array_of_instances[types]
    
def dict_of(types_from, types_to):
    types = (types_from, types_to)
    if types not in _all_dict_of_instances:
        _all_dict_of_instances[types] = _dict_of(types_from, types_to)
    return _all_dict_of_instances[types]

def ComputeThetaType(additional_types=None):
    all_types = [float, float64, int]
    if additional_types is not None:
        all_types.extend(list(additional_types))
    powerset = itertools.chain.from_iterable(itertools.combinations(all_types, r) for r in range(1, len(all_types)+1)) # equivalent to itertools.powerset without empty tuple
    theta_type = list()
    for t in powerset:
        if len(t) == 1:
            theta_type.append(tuple_of(t[0]))
        else:
            theta_type.append(tuple_of(t))
    return tuple(theta_type)
ThetaType = ComputeThetaType()
DictOfThetaType = tuple(dict_of(str, theta_subtype) for theta_subtype in ThetaType)
OnlineSizeType = (int, dict_of(str, int))

