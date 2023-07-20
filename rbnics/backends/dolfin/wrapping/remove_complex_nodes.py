# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

try:
    try:
        from ufl_legacy.algorithms.remove_complex_nodes import remove_complex_nodes
    except ImportError:
        from ufl.algorithms.remove_complex_nodes import remove_complex_nodes
except ImportError:
    def remove_complex_nodes(expr):
        return expr
