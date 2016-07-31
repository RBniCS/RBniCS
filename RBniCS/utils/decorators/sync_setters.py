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
## @file numpy_io.py
#  @brief I/O helper functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import types
from RBniCS.utils.decorators.extends import Extends
from RBniCS.utils.decorators.override import override

def SyncSetters(other_object__name, method__name, private_attribute__name):
    def SyncSetters_Decorator(Parent):
        
        @Extends(Parent, preserve_class_name=True)
        class SyncSetters_Class(Parent):
            
            @override
            def __init__(self, *args, **kwargs):
                # Call the parent initialization
                Parent.__init__(self, *args, **kwargs)
                # Get other_object
                other_object = getattr(self, other_object__name)
                # Get original methods
                self__original_method = getattr(self, method__name)
                other_object__original_method = getattr(other_object, method__name)
                # Override my setter to propagate from self to other_object
                def self__overridden_method(self, arg):
                    self__original_method(arg)
                    if getattr(other_object, private_attribute__name) is not arg:
                        other_object__original_method(arg)
                self__overridden_method = override(self__overridden_method)
                setattr(self, method__name, types.MethodType(self__overridden_method, self))
                # Override setter of other_object to propagate from other_object to self
                def other_object__overridden_method(_, arg): # _ is other_object, self is an instance of the Parent class
                    other_object__original_method(arg)
                    if getattr(self, private_attribute__name) is not arg:
                        self__original_method(arg)
                other_object__overridden_method = override(other_object__overridden_method)
                setattr(other_object, method__name, types.MethodType(other_object__overridden_method, other_object))
                # Make sure that the value of my attribute is in sync with the value the is currently 
                # stored in other_object, because it was set before overriding was carried out
                self__original_method(getattr(other_object, private_attribute__name))
        
        return SyncSetters_Class
    return SyncSetters_Decorator

