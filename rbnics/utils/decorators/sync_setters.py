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
## @file numpy_io.py
#  @brief I/O helper functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import types
from rbnics.utils.decorators.override import override

def sync_setters__internal(other_object__name, method__name, private_attribute__name, method__decorator=None):
    def sync_setters_decorator(__init__):
        
        def __synced__init__(self, *args, **kwargs):
            # Call the parent initialization
            __init__(self, *args, **kwargs)
            # Get other_object
            other_object = getattr(self, other_object__name)
            # Sync setters only if the other object is not None
            if other_object is not None:
                # Get original methods
                self__original_method = getattr(self, method__name)
                other_object__original_method = getattr(other_object, method__name)
                # Override my setter to propagate from self to other_object
                def self__overridden_method(self, arg):
                    self__original_method(arg)
                    self_attribute = getattr(self, private_attribute__name)
                    other_attribute = getattr(other_object, private_attribute__name)
                    if other_attribute is not self_attribute:
                        other_object__original_method(self_attribute)
                if method__decorator is not None:
                    self__overridden_method = method__decorator(self__overridden_method)
                self__overridden_method = override(self__overridden_method)
                setattr(self, method__name, types.MethodType(self__overridden_method, self))
                # Override setter of other_object to propagate from other_object to self
                def other_object__overridden_method(_, arg): # _ is other_object, self is an instance of the parent class
                    other_object__original_method(arg)
                    other_attribute = getattr(other_object, private_attribute__name)
                    self_attribute = getattr(self, private_attribute__name)
                    if self_attribute is not other_attribute:
                        self__original_method(other_attribute)
                if method__decorator is not None:
                    other_object__overridden_method = method__decorator(other_object__overridden_method)
                other_object__overridden_method = override(other_object__overridden_method)
                setattr(other_object, method__name, types.MethodType(other_object__overridden_method, other_object))
                # Make sure that the value of my attribute is in sync with the value the is currently 
                # stored in other_object, because it was set before overriding was carried out
                self__original_method(getattr(other_object, private_attribute__name))
        
        return __synced__init__
    return sync_setters_decorator

def sync_setters(other_object__name, method__name, private_attribute__name):
    assert method__name in ("set_final_time", "set_mu", "set_mu_range", "set_time", "set_time_step_size") # other uses have not been considered yet
    if method__name in ("set_final_time", "set_mu", "set_time", "set_time_step_size"):
        return sync_setters__internal(other_object__name, method__name, private_attribute__name)
    elif method__name == "set_mu_range":
        def set_mu_range__decorator(set_mu_range__method):
            def set_mu_range__decorated(self_, mu_range):
                try:
                    set_mu_range__method(self_, mu_range)
                except AssertionError as assertion:
                    if str(assertion) == "mu and mu_range must have the same length":
                        # This may happen because mu_range has not been set yet when
                        # called recursively across e.g. SCM, EIM, etc.
                        # Temporarily disable it, since it will be called anyway at 
                        # the last recursion step
                        original_set_mu = self_.set_mu
                        def disabled_set_mu(self_, mu):
                            pass
                        setattr(self_, "set_mu", types.MethodType(disabled_set_mu, self_))
                        # Call again set_mu_range
                        set_mu_range__method(self_, mu_range)
                        # Restore original set_mu
                        setattr(self_, "set_mu", original_set_mu)
                    else:
                        raise
            return set_mu_range__decorated
        return sync_setters__internal(other_object__name, method__name, private_attribute__name, set_mu_range__decorator)
    else:
        raise AssertionError("Invalid method in sync_setters.")
    
