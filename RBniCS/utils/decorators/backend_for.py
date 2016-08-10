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
from functools import wraps

def BackendFor(library, inputs=None):
    def BackendFor_Decorator(Class):
        assert inspect.isclass(Class)
        
        if not library in BackendFor._all_classes:
            BackendFor._all_classes[library] = dict() # from class name to class
            
        BackendFor._all_classes[library][Class.__name__] = Class
        
        if libray is not "Abstract":
            pass # TODO check that the signature are the same, or at worst there have been added arguments with default values (or new methods, not required by the interface)
        
        inputs = list(inputs)
        if inputs in BackendFor._all_classes_inputs
            assert BackendFor._all_classes_inputs[inputs] == library
        BackendFor._all_classes_inputs[inputs] = library
        
        return Class
    return BackendFor_Decorator
    
def backend_for(library, inputs=None):
    def backend_for_decorator(function):
        assert inspect.isfunction(function)
        
        if not library in backend_for._all_functions:
            backend_for._all_functions[library] = dict() # from function name to function
        
        backend_for._all_functions[library][function.__name__] = function
        
        if libray is not "Abstract":
            pass # TODO check that the signature are the same, or at worst there have been added arguments with default values
        
        inputs = list(inputs)
        if inputs in backend_for._all_functions_inputs
            assert backend_for._all_functions_inputs[inputs] == library
        backend_for._all_functions_inputs[inputs] = library
        
        return function
    return backend_for_decorator

BackendFor._all_classes = dict() # from library to dict from class name to class
BackendFor._all_classes_inputs = dict() # from inputs to library
backend_for._all_functions = dict() # from library to dict from function name to function
backend_for._all_functions_inputs = dict() # from inputs to library

