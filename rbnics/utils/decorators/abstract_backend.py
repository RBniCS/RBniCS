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

# Declare abstract vector type
from abc import ABCMeta, abstractmethod, abstractproperty
import inspect
from functools import wraps
from rbnics.utils.decorators.backend_for import BackendFor, backend_for
from rbnics.utils.decorators.extends import Extends

def AbstractBackend(Class):
    assert inspect.isclass(Class)
    assert hasattr(Class, "__metaclass__")
    assert Class.__metaclass__ is ABCMeta
    
    @BackendFor("Abstract")
    @Extends(Class, preserve_class_name=True)
    class AbstractBackend_Class(Class):
        pass
    
    return AbstractBackend_Class
    
def abstract_backend(function):
    assert inspect.isfunction(function)
    
    @backend_for("Abstract")
    @wraps(function)
    def abstract_backend_function(*args, **kwargs):
        raise NotImplementedError("This function is just a placeholder, it should never get called. If you see this error you have probably forgotten to implement a function in your backend.")
    
    return abstract_backend_function
    
def abstract_online_backend(function):
    assert inspect.isfunction(function)
    
    @backend_for("Abstract")
    @wraps(function)
    def abstract_online_backend_function(*args, **kwargs):
        raise NotImplementedError("This function is just a placeholder, it should never get called. If you see this error you have probably forgotten to implement a function in your backend, or you are trying to call this method with a backend which is not supposed to be used online.")
    
    return abstract_online_backend_function
    
def abstractonlinemethod(method):
    assert inspect.isfunction(method)
    
    @wraps(method)
    def abstractonlinemethod_function(*args, **kwargs):
        raise NotImplementedError("This method is just a placeholder, it should never get called. If you see this error you have probably forgotten to implement a method in your backend, or you are trying to call this method with a backend which is not supposed to be used online.")
    
    return abstractonlinemethod_function
    
class abstractclassmethod(classmethod):

    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable)
