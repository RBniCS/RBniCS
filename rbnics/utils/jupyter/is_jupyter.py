# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

def is_jupyter():
    try:
        from IPython import get_ipython
    except (ImportError, ModuleNotFoundError):
        return False
    else:
        ipython_module = type(get_ipython()).__module__
        return ipython_module.startswith("ipykernel.") or ipython_module.startswith("google.colab.")
