# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import importlib
import sys
import inspect

# Initialize __all__ variable
__all__ = []

# Process configuration files first
from rbnics.utils.config import config


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

    # Copy all classes and functions from (current) online backend to this module with "Online" and "online_" prefixes,
    # respectively
    for class_or_function_name in sys.modules[__name__ + "." + online_backend].__all__:
        class_or_function = getattr(sys.modules[__name__ + "." + online_backend], class_or_function_name)
        assert inspect.isclass(class_or_function) or inspect.isfunction(class_or_function)
        # The next if statement is less restrictive than inspect.isclass(class_or_function),
        # because for instance OnlineMatrix is implemented as function rather than a class
        if class_or_function_name[0].isupper():
            prefix = "Online"
        else:
            prefix = "online_"
        setattr(sys.modules[__name__], prefix + class_or_function_name, class_or_function)
        sys.modules[__name__].__all__.append(prefix + class_or_function_name)

    # Also copy the online wrapping module, without prefixes
    sys.modules[__name__ + ".wrapping"] = sys.modules[__name__ + "." + online_backend].wrapping


# Set online backend
set_online_backend(config.get("backends", "online backend"))
