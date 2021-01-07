# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

try:
    from ufl.algorithms.remove_complex_nodes import remove_complex_nodes
except ImportError:
    def remove_complex_nodes(expr):
        return expr
