# Copyright (C) 2015-2019 by the RBniCS authors
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
import sys
import inspect

# Initialize __all__ variable
__all__ = list()

# Helper function to set the online backend
def set_online_backend(online_backend):
    # Clean up previously defined online backend classes and functions
    if hasattr(sys.modules[__name__], "__all__"):
        __all__cleaned_up = list()
        for class_or_function_name in sys.modules[__name__].__all__:
            if class_or_function_name.startswith("online_") or class_or_function_name.startswith("Online"):
                delattr(sys.modules[__name__], class_or_function_name)
            else:
                __all__cleaned_up.append(class_or_function_name)
        sys.modules[__name__].__all__ = __all__cleaned_up
                
    # Import (current) online backend
    importlib.import_module(__name__ + "." + online_backend)
    importlib.import_module(__name__ + "." + online_backend + ".wrapping")

    # Copy all classes and functions from (current) online backend to this module with "Online" and "online_" prefixes, respectively
    for class_or_function_name in sys.modules[__name__ + "." + online_backend].__all__:
        class_or_function = getattr(sys.modules[__name__ + "." + online_backend], class_or_function_name)
        assert inspect.isclass(class_or_function) or inspect.isfunction(class_or_function)
        if class_or_function_name[0].isupper(): # less restrictive than inspect.isclass(class_or_function), because for instance OnlineMatrix is implemented as function rather than a class
            prefix = "Online"
        else: # less restrictive than inspect.isfunction(class_or_function)
            prefix = "online_"
        setattr(sys.modules[__name__], prefix + class_or_function_name, class_or_function)
        sys.modules[__name__].__all__.append(prefix + class_or_function_name)
    
    # Also copy the online wrapping module, without prefixes
    sys.modules[__name__ + ".wrapping"] = sys.modules[__name__ + "." + online_backend].wrapping

# Get the online backend name
from rbnics.utils.config import config
set_online_backend(config.get("backends", "online backend"))
